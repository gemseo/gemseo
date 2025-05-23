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
"""Tests for the class BaseSensitivityAnalysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import linspace
from numpy import pi
from numpy import sin
from numpy.testing import assert_array_equal

from gemseo import create_discipline
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import Discipline
from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    BaseSensitivityAnalysis,
)
from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
    FirstOrderIndicesType,
)
from gemseo.uncertainty.sensitivity.correlation_analysis import CorrelationAnalysis
from gemseo.uncertainty.sensitivity.morris_analysis import MorrisAnalysis
from gemseo.uncertainty.sensitivity.sobol_analysis import SobolAnalysis
from gemseo.utils.testing.helpers import concretize_classes
from gemseo.utils.testing.helpers import image_comparison

if TYPE_CHECKING:
    from gemseo.disciplines.analytic import AnalyticDiscipline
    from gemseo.typing import StrKeyMapping


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


class Ishigami1D(Discipline):
    """A version of the Ishigami function indexed by a 1D variable."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["x1", "x2", "x3"])
        self.io.output_grammar.update_from_names(["out"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        local_data = self.io.data
        x_1 = local_data["x1"]
        x_2 = local_data["x2"]
        x_3 = local_data["x3"]
        time = linspace(0, 1, 100)
        output = sin(x_1) + 7 * sin(x_2) ** 2 + 0.1 * x_3**4 * sin(x_1) * time
        return {"out": output}


class MockSensitivityAnalysis(BaseSensitivityAnalysis):
    """This sensitivity analysis returns dummy indices with dummy methods m1 and m2.

    It relies on a dummy dataset with inputs x1 and x2 and outputs y1 and y2. xi and yi
    of dimension i.
    """

    @dataclass
    class SensitivityIndices:
        m1: FirstOrderIndicesType
        m2: FirstOrderIndicesType

    def __init__(self) -> None:
        self.dataset = IODataset()
        data = array([[1, 2, 3]])
        self._input_names = ["x1", "x2"]
        self._output_names = ["y1", "y2"]
        self.dataset.add_group(
            self.dataset.INPUT_GROUP, data, self._input_names, {"x1": 1, "x2": 2}
        )
        self.dataset.add_group(
            self.dataset.OUTPUT_GROUP, data, ["y1", "y2"], {"y1": 1, "y2": 2}
        )

    @property
    def indices(self):
        return self.SensitivityIndices(
            m1={
                "y1": [{"x1": array([1.25]), "x2": array([1.5, 1.75])}],
                "y2": [
                    {"x1": array([2.25]), "x2": array([2.5, 2.75])},
                    {"x1": array([3.25]), "x2": array([3.5, 3.75])},
                ],
            },
            m2={
                "y1": [{"x1": array([1]), "x2": array([1, 1])}],
                "y2": [
                    {"x1": array([2]), "x2": array([2, 2])},
                    {"x1": array([3]), "x2": array([3, 3])},
                ],
            },
        )

    @property
    def main_indices(self):
        return self.indices.m1


class SecondMockSensitivityAnalysis(MockSensitivityAnalysis):
    """This sensitivity analysis uses the values of m2 as main indices."""

    @property
    def main_indices(self):
        return self.indices.m2


class MockMorrisAnalysisIndices(MorrisAnalysis):
    """A mock of a Morris sensitivity analysis, from which a dataset can be exported."""

    def __init__(self) -> None:
        self.dataset = IODataset()
        self.dataset.sizes = {
            "x1": 1,
        }

    @property
    def input_names(self) -> list[str]:
        """The names of the inputs."""
        return ["x1"]

    @property
    def indices(self):
        return self.SensitivityIndices(
            mu={
                "y": [
                    {
                        "x1": array([-0.36000398]),
                    }
                ]
            },
            mu_star={
                "y": [
                    {
                        "x1": array([0.67947346]),
                    }
                ]
            },
            sigma={
                "y": [
                    {
                        "x1": array([0.98724949]),
                    }
                ]
            },
            relative_sigma={
                "y": [
                    {
                        "x1": array([1.45296254]),
                    }
                ]
            },
            min={
                "y": [
                    {
                        "x1": array([0.0338188]),
                    }
                ]
            },
            max={
                "y": [
                    {
                        "x1": array([2.2360336]),
                    }
                ]
            },
        )


@pytest.fixture
def mock_sensitivity_analysis() -> MockSensitivityAnalysis:
    """Return an instance of MockSensitivityAnalysis."""
    with concretize_classes(MockSensitivityAnalysis):
        return MockSensitivityAnalysis()


@pytest.fixture
def second_mock_sensitivity_analysis() -> SecondMockSensitivityAnalysis:
    """Return an instance of SecondMockSensitivityAnalysis."""
    with concretize_classes(SecondMockSensitivityAnalysis):
        return SecondMockSensitivityAnalysis()


BARPLOT_TEST_PARAMETERS = {
    "without_option": ({}, ["bar_plot"]),
    "standardize": ({"outputs": "y2", "standardize": True}, ["bar_plot_standardize"]),
    "inputs": ({"outputs": "y2", "input_names": ["x1"]}, ["bar_plot_inputs"]),
    "inputs_standardize": (
        {"standardize": True, "input_names": ["x1"], "outputs": "y2"},
        ["bar_plot_inputs_standardize"],
    ),
    "outputs": ({"outputs": [("y2", 0)]}, ["bar_plot_outputs"]),
    "n_digits": ({"outputs": "y2", "n_digits": 1}, ["bar_plot_n_digits"]),
    "do_not_sort": ({"outputs": ["y1", "y2"], "sort": False}, ["bar_plot_do_not_sort"]),
    "sorting_output_as_str": (
        {"outputs": ["y1", "y2"], "sorting_output": "y2"},
        ["bar_plot_sorting_output_as_str"],
    ),
    "sorting_output_as_tuple": (
        {"outputs": ["y1", "y2"], "sorting_output": ("y2", 1)},
        ["bar_plot_sorting_output_as_tuple"],
    ),
}


@pytest.mark.parametrize(
    ("kwargs", "baseline_images"),
    BARPLOT_TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=BARPLOT_TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot_bar(kwargs, baseline_images, mock_sensitivity_analysis) -> None:
    """Check that a Barplot is created with plot_bar."""
    mock_sensitivity_analysis.plot_bar(save=False, show=False, **kwargs)


RADAR_TEST_PARAMETERS = {
    "without_option": ({}, ["radar_plot"]),
    "standardize": ({"outputs": "y2", "standardize": True}, ["radar_plot_standardize"]),
    "inputs": ({"outputs": "y2", "input_names": ["x1"]}, ["radar_plot_inputs"]),
    "inputs_standardize": (
        {"standardize": True, "input_names": ["x1"], "outputs": "y2"},
        ["radar_plot_inputs_standardize"],
    ),
    "outputs": ({"outputs": [("y2", 0)]}, ["radar_plot_outputs"]),
}


@pytest.mark.parametrize(
    ("kwargs", "baseline_images"),
    RADAR_TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=RADAR_TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot_radar(kwargs, baseline_images, mock_sensitivity_analysis) -> None:
    """Check that a RadarChart is created with plot_radar."""
    mock_sensitivity_analysis.plot_radar(save=False, show=False, **kwargs)


COMPARISON_TEST_PARAMETERS = {
    "use_bar": (True, ["comparison_bar"]),
    "use_radar": (False, ["comparison_radar"]),
}


@pytest.mark.parametrize(
    ("use_bar_plot", "baseline_images"),
    COMPARISON_TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=COMPARISON_TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot_comparison(
    use_bar_plot, baseline_images, discipline, parameter_space
) -> None:
    """Check if the comparison of sensitivity indices works."""
    spearman = CorrelationAnalysis()
    spearman.compute_samples([discipline], parameter_space, 10)
    spearman.compute_indices()
    pearson = CorrelationAnalysis()
    pearson.compute_samples([discipline], parameter_space, 10)
    pearson.main_method = pearson.Method.PEARSON
    pearson.compute_indices()
    plot = pearson.plot_comparison(
        spearman, "out", save=False, title="foo", use_bar_plot=use_bar_plot
    )
    assert plot.title == "foo"


def test_input_names(mock_sensitivity_analysis) -> None:
    """Check the value of the attribute input_names.

    Args:
        mock_sensitivity_analysis: The sensitivity analysis.
    """
    assert mock_sensitivity_analysis.input_names == ["x1", "x2"]


@pytest.mark.parametrize("output", ["y1", ("y1", 0), ("y2", 0)])
def test_sort_parameters(mock_sensitivity_analysis, output) -> None:
    """Check if the parameters are well sorted.

    Args:
        mock_sensitivity_analysis: The sensitivity analysis.
        output: The value used to sort the parameters.
    """
    parameters = mock_sensitivity_analysis.sort_input_variables(output)
    assert parameters == ["x2", "x1"]


def test_convert_to_dataset(mock_sensitivity_analysis) -> None:
    """Check if the sensitivity indices are well converted to Dataset.

    Args:
        mock_sensitivity_analysis: The sensitivity analysis.
    """
    dataset = mock_sensitivity_analysis.to_dataset()
    assert isinstance(dataset, Dataset)
    assert (dataset.index == ["x1", "x2[0]", "x2[1]"]).all()
    assert dataset.variable_names == ["y1", "y2"]
    assert dataset.group_names == ["m1", "m2"]
    assert_array_equal(
        dataset.get_view(group_names="m2", variable_names="y1").to_numpy(),
        array([[1], [1], [1]]),
    )
    assert_array_equal(
        dataset.get_view(group_names="m2", variable_names="y2").to_numpy(),
        array([[2, 3], [2, 3], [2, 3]]),
    )


@pytest.fixture(scope="module")
def ishigami() -> SobolAnalysis:
    """Return the Sobol' analysis for the Ishigami function."""
    space = ParameterSpace()
    for variable in ["x1", "x2", "x3"]:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-pi, maximum=pi
        )

    sobol_analysis = SobolAnalysis()
    sobol_analysis.compute_samples(
        [Ishigami1D()], space, 100, compute_second_order=False
    )
    sobol_analysis.main_method = "total"
    sobol_analysis.compute_indices()
    return sobol_analysis


ONE_D_FIELD_TEST_PARAMETERS = {
    "without_option": ({}, ["1d_field"], "out"),
    "without_option_with_tuple": ({}, ["1d_field"], ("out", 0)),
    "standardize": ({"standardize": True}, ["1d_field_standardize"], "out"),
    "inputs": ({"input_names": ["x1", "x3"]}, ["1d_field_inputs"], "out"),
    "inputs_standardize": (
        {"standardize": True, "input_names": ["x1", "x3"]},
        ["1d_field_inputs_standardize"],
        "out",
    ),
    "properties": ({"properties": {"xlabel": "foo"}}, ["1d_field_properties"], "out"),
}


@pytest.mark.parametrize(
    ("kwargs", "baseline_images", "output"),
    ONE_D_FIELD_TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=ONE_D_FIELD_TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot_1d_field(kwargs, baseline_images, output, ishigami) -> None:
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
        {"input_names": ["x1", "x3"]},
        [f"2d_field_inputs_{i}" for i in range(2)],
    ),
    "inputs_standardize": (
        {"standardize": True, "input_names": ["x1", "x3"]},
        [f"2d_field_inputs_standardize_{i}" for i in range(2)],
    ),
}


@pytest.mark.parametrize(
    ("kwargs", "baseline_images"),
    TWO_D_FIELD_TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TWO_D_FIELD_TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot_2d_field(kwargs, baseline_images, ishigami) -> None:
    """Check if a 2D field is well plotted with mesh."""
    times = linspace(0, 1, 10)
    mesh = array([[time1, time2] for time1 in times for time2 in times])
    ishigami.plot_field("out", save=False, show=False, mesh=mesh, **kwargs)


def test_standardize_indices() -> None:
    """Check that the method standardize_indices() works."""
    indices = {
        "y1": [{"x1": array([-2.0]), "x2": array([0.5]), "x3": array([1.0])}],
        "y2": [
            {"x1": array([2.0]), "x2": array([0.5]), "x3": array([-1.0])},
            {"x1": array([0.0]), "x2": array([0.5]), "x3": array([2.0])},
        ],
    }
    standardized_indices = BaseSensitivityAnalysis.standardize_indices(indices)
    expected_standardized_indices = {
        "y1": [{"x1": array([1.0]), "x2": array([0.25]), "x3": array([0.5])}],
        "y2": [
            {"x1": array([1.0]), "x2": array([0.25]), "x3": array([0.5])},
            {"x1": array([0.0]), "x2": array([0.25]), "x3": array([1.0])},
        ],
    }
    assert standardized_indices == expected_standardized_indices


def test_multiple_disciplines(parameter_space) -> None:
    """Test a BaseSensitivityAnalysis with multiple disciplines.

    Args:
        parameter_space: A parameter space for the analysis.
    """
    expressions = [{"y1": "x1+x3+y2"}, {"y2": "x2+x3+2*y1"}, {"f": "x3+y1+y2"}]
    d1 = create_discipline("AnalyticDiscipline", expressions=expressions[0])
    d2 = create_discipline("AnalyticDiscipline", expressions=expressions[1])
    d3 = create_discipline("AnalyticDiscipline", expressions=expressions[2])

    with concretize_classes(BaseSensitivityAnalysis):
        sensitivity_analysis = BaseSensitivityAnalysis()
        sensitivity_analysis.compute_samples(
            [d1, d2, d3], parameter_space, 5, algo="OT_MONTE_CARLO"
        )

    assert sensitivity_analysis.dataset.get_variable_names("inputs") == [
        "x1",
        "x2",
        "x3",
    ]
    assert sensitivity_analysis.dataset.get_variable_names("outputs") == [
        "f",
        "y1",
        "y2",
    ]
    assert sensitivity_analysis.dataset.n_samples == 5
