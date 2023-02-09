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

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.uncertainty.sensitivity.morris.analysis import MorrisAnalysis
from gemseo.uncertainty.sensitivity.morris.oat import _OATSensitivity
from gemseo.utils.testing import image_comparison
from numpy import allclose
from numpy import array
from numpy import pi
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

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


@pytest.fixture
def morris():
    """Morris analysis for the Ishigami function."""
    discipline = create_discipline(
        "AnalyticDiscipline",
        expressions=FUNCTION["expression"],
        name=FUNCTION["name"],
    )

    space = ParameterSpace()
    for variable in FUNCTION["variables"]:
        space.add_random_variable(**FUNCTION["distributions"][variable])

    analysis = MorrisAnalysis([discipline], space, n_samples=None)
    analysis.compute_indices()

    return analysis


def test_morris_main_indices_outputs(morris):
    """Check that all the outputs have main indices."""
    assert {"y1", "y2"} == set(morris.main_indices.keys())


@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_main_indices_outputs_content(morris, output):
    """Check that the main indices are well-formed."""
    assert len(morris.main_indices[output]) == 1
    assert set(morris.main_indices[output][0].keys()) == set(FUNCTION["variables"])


def test_morris_main_indices(morris):
    """Check that the main indices are mu_star."""
    assert morris.main_indices == morris.indices["mu_star"]


@pytest.mark.parametrize(
    "name",
    ["mu", "mu_star", "sigma", "relative_sigma", "min", "max"],
)
def test_morris_indices_outputs(morris, name):
    """Check that all the outputs have indices."""
    assert {"y1", "y2"} == set(morris.indices[name].keys())


@pytest.mark.parametrize(
    "name",
    ["mu", "mu_star", "sigma", "relative_sigma", "min", "max"],
)
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_indices_outputs_content(morris, name, output):
    """Check that all the outputs' indices are well-formed."""
    assert len(morris.indices[name][output]) == 1
    assert set(morris.indices[name][output][0].keys()) == set(FUNCTION["variables"])


@pytest.mark.parametrize("variable", FUNCTION["variables"])
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_sigma(morris, output, variable):
    """Check that sigma is positive."""
    assert morris.indices["sigma"][output][0][variable] >= 0


@pytest.mark.parametrize("variable", FUNCTION["variables"])
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_mu(morris, output, variable):
    """Check that mu_star is greater or equal to mu."""
    assert (
        morris.indices["mu_star"][output][0][variable]
        >= morris.indices["mu"][output][0][variable]
    )


@pytest.mark.parametrize("variable", FUNCTION["variables"])
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_min_max(morris, output, variable):
    """Check that the maximum is greater or equal to the minimum."""
    assert (
        morris.indices["max"][output][0][variable]
        >= morris.indices["min"][output][0][variable]
    )


@pytest.mark.parametrize("variable", FUNCTION["variables"])
@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_relative_sigma(morris, output, variable):
    """Check that the relative sigma is equal to sigma divided by mu_star."""
    relative_sigma = morris.indices["relative_sigma"][output][0][variable]
    sigma = morris.indices["sigma"][output][0][variable]
    mu_star = morris.indices["mu_star"][output][0][variable]
    assert relative_sigma == sigma / mu_star


@pytest.mark.parametrize(
    "output_name,,kwargs,baseline_images",
    [
        ("y1", {}, ["plot_y1"]),
        ("y2", {}, ["plot_y2"]),
        ("y1", {"inputs": ["x1", "x3"]}, ["plot_inputs"]),
        ("y1", {"offset": 5}, ["plot_offset"]),
        ("y1", {"lower_mu": 1}, ["plot_lower_mu"]),
        ("y1", {"lower_sigma": 0.1}, ["plot_lower_sigma"]),
    ],
)
@image_comparison(None)
def test_plot(morris, output_name, kwargs, baseline_images, pyplot_close_all):
    """Check the main visualization method."""
    morris.plot(output_name, save=False, **kwargs)


@pytest.mark.parametrize(
    "output,expected", [("y1", ["x2", "x3", "x1"]), ("y2", ["x3", "x2", "x1"])]
)
def test_morris_sort_parameters(morris, output, expected):
    """Verify that the parameters are correctly sorted."""
    assert isinstance(morris.sort_parameters(output), list)
    assert set(morris.sort_parameters(output)) == set(FUNCTION["variables"])
    assert morris.sort_parameters(output) == expected


def test_morris_with_bad_input_dimension():
    """Check that a ValueError is raised if an input dimension is not equal to 1."""
    expressions = {"y": "x1+x2"}
    discipline = create_discipline("AnalyticDiscipline", expressions=expressions)
    space = ParameterSpace()
    space.add_random_variable(
        "x1", "OTUniformDistribution", minimum=-pi, maximum=pi, size=2
    )
    space.add_random_variable("x2", "OTUniformDistribution", minimum=-pi, maximum=pi)
    with pytest.raises(ValueError, match="Each input dimension must be equal to 1."):
        MorrisAnalysis(discipline, space, n_samples=None, n_replicates=100)


def test_morris_with_nsamples():
    """Check the number of replicates when the number of samples is specified."""
    expressions = {"y": "x1+x2"}
    discipline = create_discipline("AnalyticDiscipline", expressions=expressions)
    space = ParameterSpace()
    space.add_random_variable("x1", "OTUniformDistribution", minimum=-pi, maximum=pi)
    space.add_random_variable("x2", "OTUniformDistribution", minimum=-pi, maximum=pi)
    morris = MorrisAnalysis([discipline], space, n_samples=7)
    assert morris.n_replicates == 2


@pytest.mark.parametrize("output", FUNCTION["outputs"])
def test_morris_outputs_bounds(morris, output):
    assert morris.outputs_bounds[output][0] < morris.outputs_bounds[output][1]


@pytest.fixture
def oat():
    """A One-At-a-Time (OAT) discipline."""

    def py_func(x1, x2):
        y1 = array([x1 + x2])
        y2 = array([x1 - x2, x1 - x2])
        return y1, y2

    discipline = AutoPyDiscipline(py_func)
    space = ParameterSpace()
    space.add_variable("x1", l_b=-1.0, u_b=1.0, value=0)
    space.add_variable("x2", l_b=-1.0, u_b=1.0, value=0)
    scenario = DOEScenario(
        [discipline],
        formulation="MDF",
        objective_name="y1",
        design_space=space,
    )
    scenario.add_observable("y2")

    return _OATSensitivity(scenario, space, 0.2)


def test_oat_get_io_names(oat):
    """Check the input and output names obtained from a finite difference name."""
    assert oat.get_io_names("FD!output!input") == ["output", "input"]


def test_oat_get_fd_name(oat):
    """Check the finite difference name obtained from input and output names."""
    assert oat.get_fd_name("input", "output") == "fd!output!input"


@pytest.mark.parametrize(
    "x1,x2,fd",
    [
        (0.0, 0.0, [0.4, 0.4, 0.4, -0.4]),
        (0.7, 0.0, [-0.4, 0.4, -0.4, -0.4]),
        (0.7, 0.7, [-0.4, -0.4, -0.4, 0.4]),
    ],
)
def test_oat_execute(oat, x1, x2, fd):
    """Check the execute method."""
    oat.execute({"x1": array([x1]), "x2": array([x2])})
    assert_almost_equal(oat.local_data["fd!y1!x1"], fd[0])
    assert_almost_equal(oat.local_data["fd!y1!x2"], fd[1])
    assert_almost_equal(oat.local_data["fd!y2!x1"], fd[2])
    assert_almost_equal(oat.local_data["fd!y2!x2"], fd[3])


def test_oat_bounds(oat):
    """Check the estimation of the output bounds."""
    oat.execute({"x1": array([-1.0]), "x2": array([-1.0])})
    oat.execute({"x1": array([1.0]), "x2": array([1.0])})
    expected = {"y1": [[-2.0], [2.0]], "y2": [[-0.4, -0.4], [0.4, 0.4]]}
    assert_almost_equal(oat.output_range["y1"], expected["y1"])
    assert_almost_equal(oat.output_range["y2"], expected["y2"])


@pytest.mark.parametrize("step", [-0.1, 0.0, 0.5, 0.6])
def test_oat_with_wrong_step(step):
    """Check that a ValueError is raised when the step is not in ]0,0.5[."""
    expressions = {"y": "x1+x2"}
    discipline = create_discipline("AnalyticDiscipline", expressions=expressions)
    space = ParameterSpace()
    space.add_random_variable("x1", "OTUniformDistribution", minimum=-pi, maximum=pi)
    space.add_random_variable("x2", "OTUniformDistribution", minimum=-pi, maximum=pi)

    scenario = DOEScenario(
        [discipline],
        formulation="MDF",
        objective_name="y",
        design_space=space,
    )

    expected = (
        "Relative variation step must be "
        "strictly comprised between 0 and 0.5; got {}.".format(step)
    )

    with pytest.raises(ValueError, match=expected):
        _OATSensitivity(scenario, space, step=step)


def test_normalize(morris):
    discipline = create_discipline(
        "AnalyticDiscipline",
        expressions=FUNCTION["expression"],
        name=FUNCTION["name"],
    )

    space = ParameterSpace()
    for variable in FUNCTION["variables"]:
        space.add_random_variable(**FUNCTION["distributions"][variable])

    analysis = MorrisAnalysis([discipline], space, n_samples=None)
    analysis.compute_indices(normalize=True)
    for output_name, output_value in morris.mu_.items():
        lower = analysis.outputs_bounds[output_name][0]
        upper = analysis.outputs_bounds[output_name][1]
        for input_name in output_value[0]:
            assert allclose(
                morris.mu_[output_name][0][input_name],
                analysis.mu_[output_name][0][input_name] * (upper - lower),
            )
            assert allclose(
                morris.mu_star[output_name][0][input_name],
                analysis.mu_star[output_name][0][input_name]
                * max(abs(upper), abs(lower)),
            )
            assert allclose(
                morris.sigma[output_name][0][input_name],
                analysis.sigma[output_name][0][input_name] * (upper - lower),
            )
            assert allclose(
                morris.min[output_name][0][input_name],
                analysis.min[output_name][0][input_name] * (upper - lower),
            )
            assert allclose(
                morris.max[output_name][0][input_name],
                analysis.max[output_name][0][input_name] * (upper - lower),
            )


def test_morris_multiple_disciplines():
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

    morris = MorrisAnalysis([d1, d2, d3], space, 5)
    morris.compute_indices()

    assert morris.dataset.get_names("inputs") == ["x1", "x2", "x3"]
    assert morris.dataset.get_names("outputs") == [
        "fd!f!x1",
        "fd!f!x2",
        "fd!f!x3",
        "fd!y1!x1",
        "fd!y1!x2",
        "fd!y1!x3",
        "fd!y2!x1",
        "fd!y2!x2",
        "fd!y2!x3",
    ]
    assert morris.dataset.n_samples == 5


def test_save_load(morris, tmp_wd):
    """Check saving and loading a MorrisAnalysis."""
    morris.save("foo.pkl")
    new_morris = MorrisAnalysis.load("foo.pkl")
    assert_equal(new_morris.dataset.data, morris.dataset.data)
    assert new_morris.default_output == morris.default_output
    assert new_morris.n_replicates == morris.n_replicates
    assert new_morris.outputs_bounds == morris.outputs_bounds
