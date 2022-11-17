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
"""Tests for the class SensitivityAnalysis."""
from __future__ import annotations

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.post.dataset.bars import BarPlot
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis
from gemseo.uncertainty.sensitivity.correlation.analysis import CorrelationAnalysis
from gemseo.uncertainty.sensitivity.sobol.analysis import SobolAnalysis
from gemseo.utils.testing import image_comparison
from numpy import array
from numpy import linspace
from numpy import pi
from numpy import sin
from numpy.testing import assert_array_equal


@pytest.fixture
def discipline() -> AnalyticDiscipline:
    """Return a discipline of interest."""
    return create_discipline("AnalyticDiscipline", expressions={"out": "x1+2*x2+3*x3"})


@pytest.fixture
def parameter_space() -> ParameterSpace:
    """Return the parameter space on which to evaluate the discipline."""
    space = ParameterSpace()
    for name in ["x1", "x2", "x3"]:
        space.add_random_variable(name, "OTNormalDistribution")
    return space


class Ishigami1D(MDODiscipline):
    """A version of the Ishigami function indexed by a 1D variable."""

    def __init__(self):
        super().__init__()
        self.input_grammar.update(["x1", "x2", "x3"])
        self.output_grammar.update(["out"])

    def _run(self):
        x_1, x_2, x_3 = self.get_local_data_by_name(["x1", "x2", "x3"])
        time = linspace(0, 1, 100)
        output = sin(x_1) + 7 * sin(x_2) ** 2 + 0.1 * x_3**4 * sin(x_1) * time
        self.store_local_data(out=output)


class MockSensitivityAnalysis(SensitivityAnalysis):
    """This sensitivity analysis returns dummy indices with dummy methods m1 and m2.

    It relies on a dummy dataset with inputs x1 and x2 and outputs y1 and y2. xi and yi
    of dimension i.
    """

    def __init__(self):
        self.dataset = Dataset()
        data = array([[1, 2, 3]])
        variables = ["x1", "x2"]
        sizes = {"x1": 1, "x2": 2}
        self.dataset.add_group(self.dataset.INPUT_GROUP, data, variables, sizes)
        variables = ["y1", "y2"]
        sizes = {"y1": 1, "y2": 2}
        self.dataset.add_group(self.dataset.OUTPUT_GROUP, data, variables, sizes)
        self.dataset.row_names = ["x1(0)", "x2(0)", "x2(1)"]

    @property
    def indices(self):
        return {
            "m1": {
                "y1": [{"x1": array([1.25]), "x2": array([1.5, 1.75])}],
                "y2": [
                    {"x1": array([2.25]), "x2": array([2.5, 2.75])},
                    {"x1": array([3.25]), "x2": array([3.5, 3.75])},
                ],
            },
            "m2": {
                "y1": [{"x1": array([1]), "x2": array([1, 1])}],
                "y2": [
                    {"x1": array([2]), "x2": array([2, 2])},
                    {"x1": array([3]), "x2": array([3, 3])},
                ],
            },
        }

    @property
    def main_indices(self):
        return self.indices["m1"]


class SecondMockSensitivityAnalysis(MockSensitivityAnalysis):
    """This sensitivity analysis uses the values of m2 as main indices."""

    @property
    def main_indices(self):
        return self.indices["m2"]


@pytest.fixture
def mock_sensitivity_analysis() -> MockSensitivityAnalysis:
    """Return an instance of MockSensitivityAnalysis."""
    return MockSensitivityAnalysis()


@pytest.fixture
def second_mock_sensitivity_analysis() -> SecondMockSensitivityAnalysis:
    """Return an instance of SecondMockSensitivityAnalysis."""
    return SecondMockSensitivityAnalysis()


BARPLOT_TEST_PARAMETERS = {
    "without_option": ({"outputs": "y2"}, ["bar_plot"]),
    "standardize": ({"outputs": "y2", "standardize": True}, ["bar_plot_standardize"]),
    "inputs": ({"outputs": "y2", "inputs": ["x1"]}, ["bar_plot_inputs"]),
    "inputs_standardize": (
        {"standardize": True, "inputs": ["x1"], "outputs": "y2"},
        ["bar_plot_inputs_standardize"],
    ),
    "outputs": ({"outputs": ["y1", "y2"]}, ["bar_plot_outputs"]),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    BARPLOT_TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=BARPLOT_TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot_bar(kwargs, baseline_images, mock_sensitivity_analysis, pyplot_close_all):
    """Check that a Barplot is created with plot_bar."""
    mock_sensitivity_analysis.plot_bar(save=False, show=False, **kwargs)


RADAR_TEST_PARAMETERS = {
    "without_option": ({"outputs": "y2"}, ["radar_plot"]),
    "standardize": ({"outputs": "y2", "standardize": True}, ["radar_plot_standardize"]),
    "inputs": ({"outputs": "y2", "inputs": ["x1"]}, ["radar_plot_inputs"]),
    "inputs_standardize": (
        {"standardize": True, "inputs": ["x1"], "outputs": "y2"},
        ["radar_plot_inputs_standardize"],
    ),
    "outputs": ({"outputs": ["y1", "y2"]}, ["radar_plot_outputs"]),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    RADAR_TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=RADAR_TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot_radar(
    kwargs, baseline_images, mock_sensitivity_analysis, pyplot_close_all
):
    """Check that a RadarChart is created with plot_radar."""
    mock_sensitivity_analysis.plot_radar(save=False, show=False, **kwargs)


def test_plot_comparison(discipline, parameter_space):
    """Check if the comparison of sensitivity indices works.

    Args:
        discipline: A discipline of interest.
        parameter_space: The parameter space related to this discipline.
    """
    spearman = CorrelationAnalysis([discipline], parameter_space, 10)
    spearman.compute_indices()
    pearson = CorrelationAnalysis([discipline], parameter_space, 10)
    pearson.main_method = pearson._PEARSON
    pearson.compute_indices()
    plot = pearson.plot_comparison(spearman, "out", save=False, title="foo")
    assert plot.title == "foo"
    assert isinstance(plot, BarPlot)
    plot = pearson.plot_comparison(spearman, "out", save=False, use_bar_plot=False)
    assert isinstance(plot, RadarChart)


def test_inputs_names(mock_sensitivity_analysis):
    """Check the value of the attribute input_names.

    Args:
        mock_sensitivity_analysis: The sensitivity analysis.
    """
    assert mock_sensitivity_analysis.inputs_names == ["x1", "x2"]


@pytest.mark.parametrize("output", ["y1", ("y1", 0), ("y2", 0)])
def test_sort_parameters(mock_sensitivity_analysis, output):
    """Check if the parameters are well sorted.

    Args:
        mock_sensitivity_analysis: The sensitivity analysis.
        output: The value used to sort the parameters.
    """
    parameters = mock_sensitivity_analysis.sort_parameters(output)
    assert parameters == ["x2", "x1"]


def test_convert_to_dataset(mock_sensitivity_analysis):
    """Check if the sensitivity indices are well converted to Dataset.

    Args:
        mock_sensitivity_analysis: The sensitivity analysis.
    """
    dataset = mock_sensitivity_analysis.export_to_dataset()
    assert isinstance(dataset, Dataset)
    assert dataset.row_names == ["x1(0)", "x2(0)", "x2(1)"]
    assert dataset.variables == ["y1", "y2"]
    assert dataset.groups == ["m1", "m2"]
    assert_array_equal(dataset.data["y1"], array([[1], [1], [1]]))
    assert_array_equal(dataset.data["y2"], array([[2, 3], [2, 3], [2, 3]]))


@pytest.fixture(scope="module")
def ishigami() -> SobolAnalysis:
    """Return the Sobol' analysis for the Ishigami function."""
    space = ParameterSpace()
    for variable in ["x1", "x2", "x3"]:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-pi, maximum=pi
        )

    sobol_analysis = SobolAnalysis(
        [Ishigami1D()], space, 100, compute_second_order=False
    )
    sobol_analysis.main_method = "total"
    sobol_analysis.compute_indices()
    return sobol_analysis


ONE_D_FIELD_TEST_PARAMETERS = {
    "without_option": ({}, ["1d_field"], "out"),
    "without_option_with_tuple": ({}, ["1d_field"], ("out", 0)),
    "standardize": ({"standardize": True}, ["1d_field_standardize"], "out"),
    "inputs": ({"inputs": ["x1", "x3"]}, ["1d_field_inputs"], "out"),
    "inputs_standardize": (
        {"standardize": True, "inputs": ["x1", "x3"]},
        ["1d_field_inputs_standardize"],
        "out",
    ),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images, output",
    ONE_D_FIELD_TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=ONE_D_FIELD_TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot_1d_field(kwargs, baseline_images, output, ishigami, pyplot_close_all):
    """Check if a 1D field is well plotted."""
    ishigami.plot_field(output, save=False, show=False, **kwargs)


TWO_D_FIELD_TEST_PARAMETERS_WO_MESH = {
    "without_option": ({}, ["2d_field_wo_mesh"]),
}


TWO_D_FIELD_TEST_PARAMETERS = {
    "without_option": ({}, [f"2d_field_{i}" for i in range(3)]),
    "standardize": (
        {"standardize": True},
        [f"2d_field_standardize_{i}" for i in range(3)],
    ),
    "inputs": (
        {"inputs": ["x1", "x3"]},
        [f"2d_field_inputs_{i}" for i in range(2)],
    ),
    "inputs_standardize": (
        {"standardize": True, "inputs": ["x1", "x3"]},
        [f"2d_field_inputs_standardize_{i}" for i in range(2)],
    ),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TWO_D_FIELD_TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TWO_D_FIELD_TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot_2d_field(kwargs, baseline_images, ishigami, pyplot_close_all):
    """Check if a 2D field is well plotted with mesh."""
    times = linspace(0, 1, 10)
    mesh = array([[time1, time2] for time1 in times for time2 in times])
    ishigami.plot_field("out", save=False, show=False, mesh=mesh, **kwargs)


def test_standardize_indices():
    """Check that the method standardize_indices() works."""
    indices = {
        "y1": [{"x1": array([-2.0]), "x2": array([0.5]), "x3": array([1.0])}],
        "y2": [
            {"x1": array([2.0]), "x2": array([0.5]), "x3": array([-1.0])},
            {"x1": array([0.0]), "x2": array([0.5]), "x3": array([2.0])},
        ],
    }
    standardized_indices = SensitivityAnalysis.standardize_indices(indices)
    expected_standardized_indices = {
        "y1": [{"x1": array([1.0]), "x2": array([0.25]), "x3": array([0.5])}],
        "y2": [
            {"x1": array([1.0]), "x2": array([0.25]), "x3": array([0.5])},
            {"x1": array([0.0]), "x2": array([0.25]), "x3": array([1.0])},
        ],
    }
    assert standardized_indices == expected_standardized_indices


def test_multiple_disciplines(parameter_space):
    """Test a SensitivityAnalysis with multiple disciplines.

    Args:
        parameter_space: A parameter space for the analysis.
    """
    expressions = [{"y1": "x1+x3+y2"}, {"y2": "x2+x3+2*y1"}, {"f": "x3+y1+y2"}]
    d1 = create_discipline("AnalyticDiscipline", expressions=expressions[0])
    d2 = create_discipline("AnalyticDiscipline", expressions=expressions[1])
    d3 = create_discipline("AnalyticDiscipline", expressions=expressions[2])

    sensitivity_analysis = SensitivityAnalysis([d1, d2, d3], parameter_space, 5)

    assert sensitivity_analysis.dataset.get_names("inputs") == ["x1", "x2", "x3"]
    assert sensitivity_analysis.dataset.get_names("outputs") == ["f", "y1", "y2"]
    assert sensitivity_analysis.dataset.n_samples == 5
