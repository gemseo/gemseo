# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from numpy import abs as np_abs
from numpy import allclose
from numpy import array
from numpy import isnan
from numpy import pi
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo import create_discipline
from gemseo.core.discipline import Discipline
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.doe.oat_doe.settings.oat_doe_settings import DEFAULT_STEP
from gemseo.doe.pydoe.settings.pydoe_lhs import PYDOE_LHS_Settings
from gemseo.doe.scipy.settings.mc import MC_Settings
from gemseo.space.parameter import ParameterSpace
from gemseo.uncertainty.distribution.openturns.finite_discrete_settings import (
    OTFiniteDiscreteDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.distribution.scipy.uniform_settings import (
    SPUniformDistribution_Settings,
)
from gemseo.uncertainty.sensitivity.morris import MorrisAnalysis
from gemseo.util.testing.helper import assert_exception

FUNCTION = {
    "name": "my_function",
    "expression": {"y1": "x1+100*x2+10*x3", "y2": "x1+10*x2+100*x3"},
    "variables": ["x1", "x2", "x3"],
    "outputs": ["y1", "y2"],
    "distributions": {
        name: {
            "name": name,
            "distribution_settings": OTUniformDistribution_Settings(
                minimum=0.0, maximum=1.0
            ),
        }
        for name in ["x1", "x2", "x3"]
    },
}


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """The discipline used by the main Morris analysis."""
    return create_discipline(
        "AnalyticDiscipline",
        FUNCTION["expression"],
        name=FUNCTION["name"],
    )


@pytest.fixture(scope="module")
def parameter_space() -> ParameterSpace:
    """The parameter space used by the main Morris analysis."""
    space = ParameterSpace()
    for variable in FUNCTION["variables"]:
        space.add_random_variable(**FUNCTION["distributions"][variable])
    return space


@pytest.fixture(scope="module")
def morris(discipline, parameter_space):
    """Morris analysis for the Ishigami function."""
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], parameter_space, n_samples=0)
    analysis.compute_indices()
    return analysis


@pytest.fixture(scope="module")
def morris_missing_step(discipline, parameter_space):
    """Morris analysis for the Ishigami function, with missing step."""
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], parameter_space, n_samples=0)
    analysis.compute_indices()
    del analysis.dataset.misc["step"]
    return analysis


def test_n_replicates(morris):
    """Test the n_replicates property."""
    # Reading n_replicates from the dataset.
    assert morris.n_replicates == 5
    del morris.dataset.misc["n_replicates"]
    # Computing n_replicates and writing in the dataset.
    assert morris.n_replicates == 5
    assert morris.dataset.misc["n_replicates"] == 5


def test_morris_main_indices_outputs(morris) -> None:
    """Check that all the outputs have main indices."""
    assert {"y1", "y2"} == morris.main_indices.keys()


@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_main_indices_outputs_content(morris, output) -> None:
    """Check that the main indices are well-formed."""
    assert len(morris.main_indices[output]) == 1
    assert list(morris.main_indices[output][0]) == FUNCTION["variables"]


def test_morris_main_indices(morris) -> None:
    """Check that the main indices are mu_star."""
    assert morris.main_indices == morris.indices.mu_star


@pytest.mark.parametrize(
    "name",
    ["MU", "MU_STAR", "SIGMA", "RELATIVE_SIGMA", "MIN", "MAX"],
)
def test_morris_indices_outputs(morris, name) -> None:
    """Check that all the outputs have indices."""
    assert list(getattr(morris.indices, name.lower())) == ["y1", "y2"]


@pytest.mark.parametrize(
    "name",
    ["MU", "MU_STAR", "SIGMA", "RELATIVE_SIGMA", "MIN", "MAX"],
)
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_indices_outputs_content(morris, name, output) -> None:
    """Check that all the outputs' indices are well-formed."""
    output_data = getattr(morris.indices, name.lower())[output]
    assert len(output_data) == 1
    assert list(output_data[0]) == FUNCTION["variables"]


@pytest.mark.parametrize("variable", FUNCTION["variables"])
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_sigma(morris, output, variable) -> None:
    """Check that sigma is positive."""
    assert morris.indices.sigma[output][0][variable] >= 0


@pytest.mark.parametrize("variable", FUNCTION["variables"])
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_mu(morris, output, variable) -> None:
    """Check that mu_star is greater or equal to mu."""
    assert (
        morris.indices.mu_star[output][0][variable]
        >= morris.indices.mu[output][0][variable]
    )


@pytest.mark.parametrize("variable", FUNCTION["variables"])
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_min_max(morris, output, variable) -> None:
    """Check that the maximum is greater or equal to the minimum."""
    assert (
        morris.indices.max[output][0][variable]
        >= morris.indices.min[output][0][variable]
    )


@pytest.mark.parametrize("use_elementary_effects", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("variable", FUNCTION["variables"])
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_relative_sigma(
    discipline, parameter_space, output, variable, normalize, use_elementary_effects
) -> None:
    """Check that the relative sigma is equal to sigma divided by mu_star.

    The indices are normalized by a common factor,
    and so this ratio does not depend on the normalization.
    """
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], parameter_space, n_samples=0)
    indices = analysis.compute_indices(
        normalize=normalize, use_elementary_effects=use_elementary_effects
    )
    relative_sigma = indices.relative_sigma[output][0][variable]
    sigma = indices.sigma[output][0][variable]
    mu_star = indices.mu_star[output][0][variable]
    assert_almost_equal(relative_sigma, sigma / mu_star)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "input_names": ["x1", "x3"],
            "offset": 5,
            "lower_mu": 0.02,
            "lower_sigma": 1e-15,
            "title": "Foo Bar",
        },
    ],
)
def test_plot(morris, kwargs, snapshot_matplotlib) -> None:
    """Check the main visualization method."""
    fig = morris.plot("y1", save=False, **kwargs)
    assert isinstance(fig, Figure)


@pytest.mark.parametrize("use_elementary_effects", [False, True])
def test_plot_convention(discipline, parameter_space, use_elementary_effects) -> None:
    """Check that the figure tells which convention produced the indices."""
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], parameter_space, n_samples=0)
    analysis.compute_indices(use_elementary_effects=use_elementary_effects)
    assert analysis.uses_elementary_effects is use_elementary_effects

    ax = analysis.plot("y1", save=False).axes[0]
    if use_elementary_effects:
        assert ax.get_title().endswith(" - PYDOE_LHS - elementary effects")
    else:
        assert ax.get_title().endswith(" - PYDOE_LHS - finite differences")


@pytest.fixture
def morris_with_nan_indices() -> MorrisAnalysis:
    """A Morris analysis whose input `x1` has `nan` indices.

    `x1` is a single-valued finite discrete variable,
    and so no OAT replicate moves it,
    while `x2` and `x3` are uniformly distributed.
    """
    space = ParameterSpace()
    space.add_random_variable(
        "x1", OTFiniteDiscreteDistribution_Settings(value_to_weight={1.0: 1.0})
    )
    for name in ("x2", "x3"):
        space.add_random_variable(name, OTUniformDistribution_Settings())

    analysis = MorrisAnalysis()
    analysis.compute_samples([AnalyticDiscipline({"y": "x1+2*x2+3*x3"})], space, 0)
    analysis.compute_indices(use_elementary_effects=True)
    return analysis


def test_plot_with_nan_indices(morris_with_nan_indices) -> None:
    """Check that the plot annotates the input components that have indices.

    An input component that no OAT replicate moves has `nan` elementary effects;
    its point cannot be annotated, but the points of the other components can.
    """
    indices = morris_with_nan_indices.indices
    assert isnan(indices.mu_star["y"][0]["x1"]).all()

    ax = morris_with_nan_indices.plot("y", save=False).axes[0]
    assert [annotation.get_text() for annotation in ax.texts] == ["x2", "x3"]

    # The offsets of the labels ignore the `nan` indices, and so are finite.
    positions = [annotation.get_position() for annotation in ax.texts]
    assert not isnan(positions).any()


@pytest.mark.filterwarnings("error:All-NaN slice encountered:RuntimeWarning")
def test_plot_with_only_nan_indices(morris_with_nan_indices) -> None:
    """Check that the plot is silent when no selected input component has indices.

    The offsets of the labels would reduce an all-`nan` slice in that case,
    which numpy reports as a warning.
    """
    ax = morris_with_nan_indices.plot("y", input_names=["x1"], save=False).axes[0]
    assert not ax.texts


def test_plot_missing_step(morris_missing_step, snapshot_matplotlib) -> None:
    """The plot() method does not display the step in the title when it is missing.

    This test is for compatibility reasons
    (at one time, the step was not saved in the dataset).
    """
    morris_missing_step.plot("y1", save=False)


@pytest.mark.parametrize("standardize", [False, True])
def test_plot_bar_with_nan_indices(morris_with_nan_indices, standardize) -> None:
    """Check that the bar plot leaves out the input components without indices.

    An input component that no OAT replicate moves has `nan` elementary effects,
    which the bar plot would draw as an empty bar.
    """
    dataset = morris_with_nan_indices.plot_bar(
        ["y"], standardize=standardize, save=False
    ).dataset
    assert sorted(dataset.variable_names) == ["x2", "x3"]
    assert not isnan(dataset.to_numpy()).any()


@pytest.mark.parametrize("standardize", [False, True])
def test_plot_radar_with_nan_indices(morris_with_nan_indices, standardize) -> None:
    """Check that the radar chart leaves out the input components without indices.

    An input component that no OAT replicate moves has `nan` elementary effects,
    which the radar chart would use as a non-finite radial limit.
    """
    radar_chart = morris_with_nan_indices.plot_radar(
        ["y"], standardize=standardize, save=False
    )
    assert sorted(radar_chart.dataset.variable_names) == ["x2", "x3"]
    assert not isnan([radar_chart.settings.rmin, radar_chart.settings.rmax]).any()


@pytest.mark.parametrize("plot_name", ["plot_bar", "plot_radar"])
def test_plot_without_any_index(morris_with_nan_indices, plot_name, snapshot) -> None:
    """Check the error raised when no selected input component has indices."""
    with assert_exception(ValueError, snapshot):
        getattr(morris_with_nan_indices, plot_name)(
            ["y"], input_names=["x1"], save=False
        )


def test_plot_comparison_with_nan_indices(morris_with_nan_indices) -> None:
    """Check that an input component without indices does not spoil the comparison.

    The indices of an analysis are divided by its largest index,
    which the `nan` elementary effects of `x1` must not make `nan` too.
    The comparison is made with the finite differences of the same samples,
    which are numbers for every input component.
    """
    finite_differences = deepcopy(morris_with_nan_indices)
    finite_differences.compute_indices()
    dataset = morris_with_nan_indices.plot_comparison(
        finite_differences, "y", save=False
    ).dataset
    assert dataset.get_columns() == ["x3", "x2", "x1"]

    data = dataset.to_numpy()
    assert_equal(isnan(data), array([[False, False, True], [False] * 3]))
    assert_almost_equal(data[0, :2], [1.0, 2.0 / 3.0])


def test_sort_input_variables_with_nan_indices(morris_with_nan_indices) -> None:
    """Check that an input variable without indices is ranked last.

    The elementary effects of `x1` are `nan`,
    and so its cumulative index cannot be compared with those of `x2` and `x3`.
    """
    assert morris_with_nan_indices.sort_input_variables("y") == ["x3", "x2", "x1"]


@pytest.mark.parametrize(
    ("output", "expected"), [("y1", ["x2", "x3", "x1"]), ("y2", ["x3", "x2", "x1"])]
)
def test_morris_sort_parameters(morris, output, expected) -> None:
    """Verify that the parameters are correctly sorted."""
    assert isinstance(morris.sort_input_variables(output), list)
    assert set(morris.sort_input_variables(output)) == set(FUNCTION["variables"])
    assert morris.sort_input_variables(output) == expected


def test_morris_with_nsamples() -> None:
    """Check the number of replicates when the number of samples is specified."""
    expressions = {"y": "x1+x2"}
    discipline = create_discipline("AnalyticDiscipline", expressions)
    space = ParameterSpace()
    space.add_random_variable(
        "x1", OTUniformDistribution_Settings(minimum=-pi, maximum=pi)
    )
    space.add_random_variable(
        "x2", OTUniformDistribution_Settings(minimum=-pi, maximum=pi)
    )
    morris = MorrisAnalysis()
    morris.compute_samples([discipline], space, n_samples=7)
    assert morris.n_replicates == 2


@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_outputs_bounds(morris, output) -> None:
    assert morris.outputs_bounds[output][0] < morris.outputs_bounds[output][1]


def test_normalize(morris) -> None:
    discipline = create_discipline(
        "AnalyticDiscipline",
        FUNCTION["expression"],
        name=FUNCTION["name"],
    )

    space = ParameterSpace()
    for variable in FUNCTION["variables"]:
        space.add_random_variable(**FUNCTION["distributions"][variable])

    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], space, n_samples=0)
    analysis.compute_indices(normalize=True)
    for output_name, output_value in morris.indices.mu.items():
        lower = analysis.outputs_bounds[output_name][0]
        upper = analysis.outputs_bounds[output_name][1]
        for input_name in output_value[0]:
            assert allclose(
                morris.indices.mu[output_name][0][input_name],
                analysis.indices.mu[output_name][0][input_name] * (upper - lower),
            )
            assert allclose(
                morris.indices.mu_star[output_name][0][input_name],
                analysis.indices.mu_star[output_name][0][input_name] * (upper - lower),
            )
            assert allclose(
                morris.indices.sigma[output_name][0][input_name],
                analysis.indices.sigma[output_name][0][input_name] * (upper - lower),
            )
            assert allclose(
                morris.indices.min[output_name][0][input_name],
                analysis.indices.min[output_name][0][input_name] * (upper - lower),
            )
            assert allclose(
                morris.indices.max[output_name][0][input_name],
                analysis.indices.max[output_name][0][input_name] * (upper - lower),
            )
            assert allclose(
                morris.indices.relative_sigma[output_name][0][input_name],
                analysis.indices.relative_sigma[output_name][0][input_name],
            )


def test_morris_multiple_disciplines() -> None:
    """Test the Morris Analysis for more than one discipline."""
    expressions = [{"y1": "x1+x3+y2"}, {"y2": "x2+x3+2*y1"}, {"f": "x3+y1+y2"}]
    d1 = create_discipline("AnalyticDiscipline", expressions[0])
    d2 = create_discipline("AnalyticDiscipline", expressions[1])
    d3 = create_discipline("AnalyticDiscipline", expressions[2])

    space = ParameterSpace()

    for variable in ["x1", "x2", "x3"]:
        space.add_random_variable(
            variable, OTUniformDistribution_Settings(minimum=-10, maximum=10)
        )

    morris = MorrisAnalysis()
    morris.compute_samples([d1, d2, d3], space, 5)
    morris.compute_indices()

    assert morris.dataset.get_variable_names("inputs") == ["x1", "x2", "x3"]
    assert morris.dataset.get_variable_names("outputs") == ["f", "y1", "y2"]
    assert morris.dataset.n_samples == 1 + 3


@pytest.mark.parametrize(("n_samples", "expected_n_samples"), [(0, 20), (8, 8), (9, 8)])
def test_n_samples(
    discipline,
    parameter_space,
    n_samples,
    expected_n_samples,
    enable_discipline_statistics,
) -> None:
    """Check the effect of n_samples."""
    n_calls = discipline.execution_statistics.n_executions
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], parameter_space, n_samples=n_samples)
    assert len(analysis.dataset) == expected_n_samples
    assert discipline.execution_statistics.n_executions - n_calls == expected_n_samples


def test_algo_settings(discipline, parameter_space) -> None:
    """Check the effect of algo_settings."""
    analysis = MorrisAnalysis()
    analysis.compute_samples(
        [discipline],
        parameter_space,
        n_samples=0,
    )
    reference = analysis.dataset
    analysis = MorrisAnalysis()
    analysis.compute_samples(
        [discipline], parameter_space, n_samples=0, algo_settings=MC_Settings()
    )
    assert not reference.equals(analysis.dataset)


def test_compute_indices_output_names(morris) -> None:
    """Check compute_indices with different types for output_names."""
    assert morris.compute_indices(["y1"]).mu
    assert morris.compute_indices("y1").mu
    # morris is a module-scoped fixture and so the original indexes must be restored.
    morris.compute_indices()


def test_too_few_samples(discipline, parameter_space, snapshot) -> None:
    """Check that the MorrisAnalysis raises a ValueError is n_samples is too small."""
    analysis = MorrisAnalysis()
    with assert_exception(ValueError, snapshot):
        analysis.compute_samples([discipline], parameter_space, n_samples=2)


def test_output_names() -> None:
    """Check that the argument output_names is correctly taken into account.

    See https://gitlab.com/gemseo/dev/gemseo/-/issues/866
    """
    discipline = AnalyticDiscipline({"y": "x", "z": "x"})
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("x", SPUniformDistribution_Settings())
    sensitivity_analysis = MorrisAnalysis()
    sensitivity_analysis.compute_samples(
        disciplines=[discipline],
        parameter_space=parameter_space,
        n_samples=0,
        output_names=["y"],
    )
    sensitivity_analysis.compute_indices()
    mu_ = sensitivity_analysis.indices.mu
    assert_almost_equal(mu_["y"][0]["x"], array([0.05]))
    assert "z" not in mu_


def test_log(caplog, discipline, parameter_space, enable_discipline_statistics) -> None:
    """Check the log generated by a Morris analysis."""
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], parameter_space, 4)
    result = "\n".join([line[2] for line in caplog.record_tuples])
    pattern = r"""^\*\*\* Start MorrisAnalysisSamplingPhase execution \*\*\*
MorrisAnalysisSamplingPhase
   Disciplines: my_function
   MDO formulation: MDF
Evaluation problem:
   Evaluate the functions: y1, y2
   over the design space:
      \+------\+-------------------------------\+
      \| Name \|          Distribution         \|
      \+------\+-------------------------------\+
      \|  x1  \| Uniform\(lower=0\.0, upper=1\.0\) \|
      \|  x2  \| Uniform\(lower=0\.0, upper=1\.0\) \|
      \|  x3  \| Uniform\(lower=0\.0, upper=1\.0\) \|
      \+------\+-------------------------------\+
Running the algorithm MorrisDOE:
    25%\|██▌       \| 1\/4 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
    50%\|█████     \| 2\/4 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
    75%\|███████▌  \| 3\/4 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
   100%\|██████████\| 4\/4 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
\*\*\* End MorrisAnalysisSamplingPhase execution \(time: \d+:\d+:\d+\.\d+\) \*\*\*$"""
    assert re.match(pattern, result)


def test_n_replicates_error(snapshot):
    """Check that the property n_replicates cannot be used without a dataset."""
    analysis = MorrisAnalysis()
    with assert_exception(ValueError, snapshot):
        analysis.n_replicates


def test_from_samples(morris, tmp_wd):
    """Check the instantiation from samples."""
    file_path = Path("samples.pkl")
    morris.dataset.to_pickle(file_path)
    new_morris = MorrisAnalysis(samples=file_path)
    new_morris.compute_indices()
    assert new_morris.indices == morris.indices


@pytest.mark.parametrize("normalize", [False, True])
def test_constant_output(discipline_with_constant_output_and_space, normalize):
    """Check that MorrisAnalysis supports constant outputs."""
    discipline, uncertain_space = discipline_with_constant_output_and_space
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 0)
    indices = analysis.compute_indices(normalize=normalize)
    assert indices.mu["constant"][0] is None
    assert indices.mu["varying"][0] is not None


def test_morris_vectorial_input(snapshot_matplotlib):
    """Check that the Morris plot for vectorial input correctly labels the input
    components."""

    class MyDisc(Discipline):
        def __init__(self):
            super().__init__()
            self.io.input_grammar.update_from_names(["x1"])
            self.io.output_grammar.update_from_names(["y1"])

        def _run(self, input_data):
            return {"y1": input_data["x1"]}

    discipline = MyDisc()
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable(
        "x1", OTUniformDistribution_Settings(minimum=-pi, maximum=pi), size=2
    )
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], uncertain_space, 0)
    analysis.compute_indices(normalize=True)
    analysis.plot(save=False, output="y1")


def test_morris_all_replicates() -> None:
    """Check that the indices are computed from all the OAT replicates.

    See https://gitlab.com/gemseo/dev/gemseo/-/work_items/1894
    """
    space = ParameterSpace()
    for name in ("x1", "x2"):
        space.add_random_variable(
            name, OTUniformDistribution_Settings(minimum=1.0, maximum=2.0)
        )

    analysis = MorrisAnalysis()
    analysis.compute_samples([AnalyticDiscipline({"y": "x1+2*x2"})], space, 0)
    indices = analysis.compute_indices()

    dataset = analysis.dataset
    output_data = dataset.get_view(group_names=dataset.OUTPUT_GROUP).to_numpy()
    stride = space.dimension + 1
    for index, name in enumerate(space):
        differences = output_data[index + 1 :: stride] - output_data[index::stride]
        assert len(differences) == analysis.n_replicates
        assert_almost_equal(indices.mu["y"][0][name], differences.mean(0))
        assert_almost_equal(indices.mu_star["y"][0][name], np_abs(differences).mean(0))
        assert_almost_equal(
            indices.sigma["y"][0][name], differences.var(0, ddof=1) ** 0.5
        )
        assert_almost_equal(indices.min["y"][0][name], np_abs(differences).min(0))
        assert_almost_equal(indices.max["y"][0][name], np_abs(differences).max(0))


def test_morris_sigma_unbiased() -> None:
    """Check that sigma is the unbiased standard deviation.

    The variance is divided by $R-1$, as in Morris (1991),
    and is zero when there is a single OAT replicate.
    """
    space = ParameterSpace()
    for name in ("x1", "x2"):
        space.add_random_variable(
            name, OTUniformDistribution_Settings(minimum=1.0, maximum=2.0)
        )

    discipline = AnalyticDiscipline({"y": "(x1-1.5)**2*x2"})

    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], space, 0, n_replicates=1)
    assert analysis.n_replicates == 1
    assert_equal(analysis.compute_indices().sigma["y"][0]["x1"], array([0.0]))

    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], space, 0, n_replicates=4)
    indices = analysis.compute_indices()
    dataset = analysis.dataset
    output_data = dataset.get_view(group_names=dataset.OUTPUT_GROUP).to_numpy()
    stride = space.dimension + 1
    differences = output_data[1::stride] - output_data[0::stride]
    sigma = indices.sigma["y"][0]["x1"]
    assert_almost_equal(sigma, differences.var(0, ddof=1) ** 0.5)
    assert not allclose(sigma, differences.var(0, ddof=0) ** 0.5)


@pytest.mark.parametrize("step", [DEFAULT_STEP, 0.1])
def test_morris_elementary_effects(step) -> None:
    """Check the indices computed from the elementary effects.

    An elementary effect is a finite difference divided by the step that produced it.
    The input variables are uniformly distributed with different ranges, and so have
    constant but different steps: the finite differences are the increments of the
    output while the elementary effects are its partial derivatives, whatever the
    relative step.
    """
    space = ParameterSpace()
    space.add_random_variable(
        "x1", OTUniformDistribution_Settings(minimum=1.0, maximum=2.0)
    )
    space.add_random_variable(
        "x2", OTUniformDistribution_Settings(minimum=0.0, maximum=10.0)
    )

    analysis = MorrisAnalysis()
    analysis.compute_samples(
        [AnalyticDiscipline({"y": "x1+2*x2"})], space, 0, step=step
    )
    r = analysis.n_replicates
    assert_almost_equal(np_abs(analysis._steps), array([[step] * r, [10 * step] * r]))

    indices = analysis.compute_indices()
    assert_almost_equal(indices.mu_star["y"][0]["x1"], array([step]))
    assert_almost_equal(indices.mu_star["y"][0]["x2"], array([2 * 10 * step]))

    effect_indices = analysis.compute_indices(use_elementary_effects=True)
    assert_almost_equal(effect_indices.mu_star["y"][0]["x1"], array([1.0]))
    assert_almost_equal(effect_indices.mu_star["y"][0]["x2"], array([2.0]))


def test_morris_elementary_effects_downward_step() -> None:
    """Check the elementary effects when the OAT step is taken downwards.

    The OAT method subtracts the step
    near the upper end of the probability scale of an input variable,
    and so this step is signed;
    an elementary effect divides a finite difference by this signed step
    and so estimates the same derivative in both directions.
    The model is linear,
    hence the elementary effects are the exact partial derivatives,
    `mu` is equal to `mu_star` and `sigma` is zero.
    """
    space = ParameterSpace()
    for name in ("x1", "x2"):
        space.add_random_variable(
            name, OTUniformDistribution_Settings(minimum=0.0, maximum=1.0)
        )

    analysis = MorrisAnalysis()
    analysis.compute_samples(
        [AnalyticDiscipline({"y": "3*x1+5*x2"})],
        space,
        0,
        n_replicates=round(1 / DEFAULT_STEP),
    )
    # The LHS puts a single initial point in the last stratum [1-DEFAULT_STEP, 1).
    assert (analysis._steps[0] < 0.0).sum() == 1

    indices = analysis.compute_indices(use_elementary_effects=True)
    assert_almost_equal(indices.mu["y"][0]["x1"], array([3.0]))
    assert_almost_equal(indices.mu_star["y"][0]["x1"], array([3.0]))
    assert_almost_equal(indices.sigma["y"][0]["x1"], array([0.0]))


def test_morris_elementary_effects_at_the_unit_upper_bound() -> None:
    """Check the elementary effects when an OAT step would reach the coordinate 1.

    A centred LHS puts an initial point at `(k+0.5)/R`,
    so `R=10` and the default step place a coordinate at `0.95`,
    for which adding the step gives exactly 1
    and the quantile function of a standard normal variable
    returns its numerical bound.
    The OAT method takes the step downwards in that case,
    hence the steps remain of the order of the relative step
    and the elementary effects of a linear model
    remain the exact partial derivatives.
    """
    space = ParameterSpace()
    for name in ("x1", "x2"):
        space.add_random_variable(name, OTNormalDistribution_Settings())

    analysis = MorrisAnalysis()
    analysis.compute_samples(
        [AnalyticDiscipline({"y": "3*x1+5*x2"})],
        space,
        0,
        n_replicates=round(1 / DEFAULT_STEP / 2),
        algo_settings=PYDOE_LHS_Settings(criterion="center", seed=1),
    )
    steps = analysis._steps
    assert (steps[0] < 0.0).sum() == 1
    assert np_abs(steps).max() < 1.0

    indices = analysis.compute_indices(use_elementary_effects=True)
    assert_almost_equal(indices.mu["y"][0]["x1"], array([3.0]))
    assert_almost_equal(indices.mu["y"][0]["x2"], array([5.0]))
    assert_almost_equal(indices.sigma["y"][0]["x1"], array([0.0]))


def test_morris_elementary_effects_non_uniform_inputs() -> None:
    """Check the elementary effects when the input variables are not uniform.

    The relative step of the OAT method is a step on the probability scale,
    and so the step of a non-uniformly distributed input variable
    changes from one OAT replicate to another.
    The model is linear,
    hence the elementary effects are the exact partial derivatives
    whatever the replicate.
    """
    space = ParameterSpace()
    for name in ("x1", "x2"):
        space.add_random_variable(name, OTNormalDistribution_Settings())

    analysis = MorrisAnalysis()
    analysis.compute_samples(
        [AnalyticDiscipline({"y": "3*x1+5*x2"})], space, 0, n_replicates=6
    )
    steps = np_abs(analysis._steps[0])
    assert steps.min() < steps.max()

    indices = analysis.compute_indices(use_elementary_effects=True)
    assert_almost_equal(indices.mu["y"][0]["x1"], array([3.0]))
    assert_almost_equal(indices.sigma["y"][0]["x1"], array([0.0]))


def test_morris_elementary_effects_zero_step(caplog) -> None:
    """Check the elementary effects when an OAT step is zero.

    The quantile function of a finite discrete random variable is flat
    over most of the probability scale,
    and so the OAT method does not move this input in most of the replicates.
    Such a replicate carries no information about the derivative
    and is left out of the elementary effects,
    whereas the finite differences keep it.

    A centred LHS puts the initial coordinates at the centres of the strata,
    which pins the number of surviving replicates whatever the seed:
    with `R` strata of width `1/R = DEFAULT_STEP`
    and a quantile function jumping at 0.25, 0.5 and 0.75,
    exactly three of these centres are followed by a jump.
    """
    space = ParameterSpace()
    space.add_random_variable(
        "x1",
        OTFiniteDiscreteDistribution_Settings(
            value_to_weight={0.0: 1.0, 10.0: 1.0, 20.0: 1.0, 30.0: 1.0}
        ),
    )
    space.add_random_variable(
        "x2", OTUniformDistribution_Settings(minimum=0.0, maximum=1.0)
    )

    n_replicates = round(1 / DEFAULT_STEP)
    analysis = MorrisAnalysis()
    analysis.compute_samples(
        [AnalyticDiscipline({"y": "x1+2*x2"})],
        space,
        0,
        n_replicates=n_replicates,
        algo_settings=PYDOE_LHS_Settings(criterion="center", seed=1),
    )
    steps = analysis._steps[0]
    assert (steps != 0.0).sum() == 3
    assert_almost_equal(np_abs(steps[steps != 0.0]), 10.0)

    # The model is linear, hence the surviving replicates give the exact derivative.
    indices = analysis.compute_indices(use_elementary_effects=True)
    assert_almost_equal(indices.mu["y"][0]["x1"], array([1.0]))
    assert_almost_equal(indices.mu_star["y"][0]["x1"], array([1.0]))
    assert_almost_equal(indices.sigma["y"][0]["x1"], array([0.0]))

    # The log says how many replicates the indices of `x1` rest on.
    assert (
        f"{n_replicates - 3} of the {n_replicates} OAT replicates "
        "do not move the input component x1; "
        "its indices computed from the elementary effects rest on the others."
    ) in caplog.text

    # A finite difference of `x1` is its step, zero replicates included.
    differences = analysis.compute_indices()
    assert_almost_equal(
        differences.mu_star["y"][0]["x1"], array([np_abs(steps).mean()])
    )


def test_morris_elementary_effects_without_step(caplog) -> None:
    """Check the elementary effects of an input that no OAT replicate moves."""
    space = ParameterSpace()
    space.add_random_variable(
        "x1", OTFiniteDiscreteDistribution_Settings(value_to_weight={1.0: 1.0})
    )
    space.add_random_variable(
        "x2", OTUniformDistribution_Settings(minimum=0.0, maximum=1.0)
    )

    analysis = MorrisAnalysis()
    analysis.compute_samples([AnalyticDiscipline({"y": "x1+2*x2"})], space, 0)
    assert_almost_equal(analysis._steps[0], 0.0)

    indices = analysis.compute_indices(use_elementary_effects=True)
    for statistic in ("mu", "mu_star", "sigma", "relative_sigma", "min", "max"):
        assert isnan(getattr(indices, statistic)["y"][0]["x1"]).all()

    assert_almost_equal(indices.mu["y"][0]["x2"], array([2.0]))
    assert (
        "The input component x1 does not vary in any OAT replicate; "
        "its indices computed from the elementary effects are NaN."
    ) in caplog.text


def test_uses_elementary_effects_after_a_failed_call() -> None:
    """Check that the convention flag follows the indices, not the call that failed.

    A call to `compute_indices` that raises leaves the previous indices in place,
    and so must leave the convention that produced them in place too.
    """
    space = ParameterSpace()
    for name in ("x1", "x2"):
        space.add_random_variable(name, OTUniformDistribution_Settings())

    analysis = MorrisAnalysis()
    analysis.compute_samples([AnalyticDiscipline({"y": "x1+2*x2"})], space, 0)
    indices = analysis.compute_indices(use_elementary_effects=True)
    assert analysis.uses_elementary_effects

    with pytest.raises(KeyError):
        analysis.compute_indices("not_an_output")

    assert analysis.uses_elementary_effects
    assert analysis.indices is indices

    title = analysis.plot("y", save=False).axes[0].get_title()
    assert title.endswith(" - elementary effects")


def test_morris_input_without_effect() -> None:
    """Check the indices of an input variable having no effect on an output.

    The output `y` is not constant
    and so its indices are not set to `None`;
    only `x3` does not reach it.

    See https://gitlab.com/gemseo/dev/gemseo/-/work_items/1895
    """
    space = ParameterSpace()
    for name in ("x1", "x2", "x3"):
        space.add_random_variable(name, OTUniformDistribution_Settings())

    analysis = MorrisAnalysis()
    analysis.compute_samples(
        [AnalyticDiscipline({"y": "x1+2*x2", "z": "x3"})], space, 0
    )
    indices = analysis.compute_indices()

    assert_equal(indices.mu_star["y"][0]["x3"], array([0.0]))
    assert_equal(indices.relative_sigma["y"][0]["x3"], array([0.0]))


def test_morris_sigma_of_signed_differences() -> None:
    """Check that sigma is the standard deviation of the signed finite differences.

    `x1` enters the model quadratically,
    and so its finite differences change sign
    and the standard deviations of the signed and absolute ones differ.

    See https://gitlab.com/gemseo/dev/gemseo/-/work_items/1896
    """
    space = ParameterSpace()
    for name in ("x1", "x2"):
        space.add_random_variable(
            name, OTUniformDistribution_Settings(minimum=1.0, maximum=2.0)
        )

    analysis = MorrisAnalysis()
    analysis.compute_samples([AnalyticDiscipline({"y": "(x1-1.5)**2*x2"})], space, 0)
    indices = analysis.compute_indices()

    dataset = analysis.dataset
    output_data = dataset.get_view(group_names=dataset.OUTPUT_GROUP).to_numpy()
    stride = space.dimension + 1
    differences = output_data[1::stride] - output_data[0::stride]
    absolute_differences = np_abs(differences)
    assert (differences < 0).any()
    assert_almost_equal(indices.sigma["y"][0]["x1"], differences.var(0, ddof=1) ** 0.5)
    assert not allclose(
        indices.sigma["y"][0]["x1"], absolute_differences.var(0, ddof=1) ** 0.5
    )
