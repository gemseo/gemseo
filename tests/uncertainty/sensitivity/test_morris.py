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
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from numpy import allclose
from numpy import array
from numpy import pi
from numpy.testing import assert_almost_equal

from gemseo import create_discipline
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.uncertainty.sensitivity.morris_analysis import MorrisAnalysis
from gemseo.utils.testing.helpers import image_comparison

FUNCTION = {
    "name": "my_function",
    "expression": {"y1": "x1+100*x2+10*x3", "y2": "x1+10*x2+100*x3"},
    "variables": ["x1", "x2", "x3"],
    "outputs": ["y1", "y2"],
    "distributions": {
        name: {
            "name": name,
            "distribution": "OTUniformDistribution",
            "minimum": 0,
            "maximum": 1,
        }
        for name in ["x1", "x2", "x3"]
    },
}


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """The discipline used by the main Morris analysis."""
    return create_discipline(
        "AnalyticDiscipline",
        expressions=FUNCTION["expression"],
        name=FUNCTION["name"],
    )


@pytest.fixture(scope="module")
def parameter_space() -> ParameterSpace:
    """The parameter space used by the main Morris analysis."""
    space = ParameterSpace()
    for variable in FUNCTION["variables"]:
        space.add_random_variable(**FUNCTION["distributions"][variable])
    return space


@pytest.fixture
def morris(discipline, parameter_space):
    """Morris analysis for the Ishigami function."""
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], parameter_space, n_samples=0)
    analysis.compute_indices()
    return analysis


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


@pytest.mark.parametrize("variable", FUNCTION["variables"])
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_relative_sigma(morris, output, variable) -> None:
    """Check that the relative sigma is equal to sigma divided by mu_star."""
    relative_sigma = morris.indices.relative_sigma[output][0][variable]
    sigma = morris.indices.sigma[output][0][variable]
    mu_star = morris.indices.mu_star[output][0][variable]
    assert relative_sigma == sigma / mu_star


@pytest.mark.parametrize(
    ("output_name", "kwargs", "baseline_images"),
    [
        ("y1", {}, ["plot_y1"]),
        ("y2", {}, ["plot_y2"]),
        ("y1", {"input_names": ["x1", "x3"]}, ["plot_inputs"]),
        ("y1", {"offset": 5}, ["plot_offset"]),
        ("y1", {"lower_mu": 1}, ["plot_lower_mu"]),
        ("y1", {"lower_sigma": 0.1}, ["plot_lower_sigma"]),
    ],
)
@image_comparison(None)
def test_plot(morris, output_name, kwargs, baseline_images) -> None:
    """Check the main visualization method."""
    fig = morris.plot(output_name, save=False, **kwargs)
    assert isinstance(fig, Figure)


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
    discipline = create_discipline("AnalyticDiscipline", expressions=expressions)
    space = ParameterSpace()
    space.add_random_variable("x1", "OTUniformDistribution", minimum=-pi, maximum=pi)
    space.add_random_variable("x2", "OTUniformDistribution", minimum=-pi, maximum=pi)
    morris = MorrisAnalysis()
    morris.compute_samples([discipline], space, n_samples=7)
    assert morris.n_replicates == 2


@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_outputs_bounds(morris, output) -> None:
    assert morris.outputs_bounds[output][0] < morris.outputs_bounds[output][1]


def test_normalize(morris) -> None:
    discipline = create_discipline(
        "AnalyticDiscipline",
        expressions=FUNCTION["expression"],
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
                analysis.indices.mu_star[output_name][0][input_name]
                * max(abs(upper), abs(lower)),
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


def test_morris_multiple_disciplines() -> None:
    """Test the Morris Analysis for more than one discipline."""
    expressions = [{"y1": "x1+x3+y2"}, {"y2": "x2+x3+2*y1"}, {"f": "x3+y1+y2"}]
    d1 = create_discipline("AnalyticDiscipline", expressions=expressions[0])
    d2 = create_discipline("AnalyticDiscipline", expressions=expressions[1])
    d3 = create_discipline("AnalyticDiscipline", expressions=expressions[2])

    space = ParameterSpace()

    for variable in ["x1", "x2", "x3"]:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-10, maximum=10
        )

    morris = MorrisAnalysis()
    morris.compute_samples([d1, d2, d3], space, 5)
    morris.compute_indices()

    assert morris.dataset.get_variable_names("inputs") == ["x1", "x2", "x3"]
    assert morris.dataset.get_variable_names("outputs") == ["f", "y1", "y2"]
    assert morris.dataset.n_samples == 1 + 3


@pytest.mark.parametrize(("n_samples", "expected_n_samples"), [(0, 20), (8, 8), (9, 8)])
def test_n_samples(discipline, parameter_space, n_samples, expected_n_samples) -> None:
    """Check the effect of n_samples."""
    n_calls = discipline.execution_statistics.n_executions
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], parameter_space, n_samples=n_samples)
    assert len(analysis.dataset) == expected_n_samples
    assert discipline.execution_statistics.n_executions - n_calls == expected_n_samples


def test_compute_indices_output_names(morris) -> None:
    """Check compute_indices with different types for output_names."""
    assert morris.compute_indices(["y1"]).mu
    assert morris.compute_indices("y1").mu


def test_too_few_samples(discipline, parameter_space) -> None:
    """Check that the MorrisAnalysis raises a ValueError is n_samples is too small."""
    analysis = MorrisAnalysis()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of samples (2) must be "
            "at least equal to the dimension of the input space plus one (3+1=4)."
        ),
    ):
        analysis.compute_samples([discipline], parameter_space, n_samples=2)


def test_output_names() -> None:
    """Check that the argument output_names is correctly taken into account.

    See https://gitlab.com/gemseo/dev/gemseo/-/issues/866
    """
    discipline = AnalyticDiscipline({"y": "x", "z": "x"})
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable(name="x", distribution="SPUniformDistribution")
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


def test_log(caplog, discipline, parameter_space) -> None:
    """Check the log generated by a Morris analysis."""
    analysis = MorrisAnalysis()
    analysis.compute_samples([discipline], parameter_space, 4)
    result = "\n".join([line[2] for line in caplog.record_tuples])
    pattern = r"""^No coupling in MDA, switching chain_linearize to True\.
\*\*\* Start MorrisAnalysisSamplingPhase execution \*\*\*
MorrisAnalysisSamplingPhase
   Disciplines: my_function
   MDO formulation: MDF
Running the algorithm MorrisDOE:
    25%\|██▌       \| 1\/4 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
    50%\|█████     \| 2\/4 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
    75%\|███████▌  \| 3\/4 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
   100%\|██████████\| 4\/4 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
\*\*\* End MorrisAnalysisSamplingPhase execution \(time: \d+:\d+:\d+\.\d+\) \*\*\*$"""
    assert re.match(pattern, result)


def test_n_replicates_error():
    """Check that the property n_replicates cannot be used without a dataset."""
    analysis = MorrisAnalysis()
    msg = (
        "There is not dataset attached to the MorrisAnalysis; "
        "please provide samples at instantiation or use compute_samples."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        analysis.n_replicates


def test_from_samples(morris, tmp_wd):
    """Check the instantiation from samples."""
    file_path = Path("samples.pkl")
    morris.dataset.to_pickle(file_path)
    new_morris = MorrisAnalysis(samples=file_path)
    new_morris.compute_indices()
    assert new_morris.indices == morris.indices
