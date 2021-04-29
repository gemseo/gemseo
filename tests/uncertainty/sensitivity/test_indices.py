# -*- coding: utf-8 -*-
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

"""Tests for the class SensitivityIndices."""

from __future__ import absolute_import, division, unicode_literals

import pytest
from numpy import array, linspace, pi, sin
from numpy.testing import assert_array_equal

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.post.dataset.bars import BarPlot
from gemseo.post.dataset.curves import Curves
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.post.dataset.surfaces import Surfaces
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis
from gemseo.uncertainty.sensitivity.correlation import CorrelationAnalysis
from gemseo.uncertainty.sensitivity.sobol import SobolAnalysis


@pytest.fixture
def discipline():  # type: (...) -> AnalyticDiscipline
    """Return a discipline of interest."""
    expressions = {"y": "x1+2*x2+3*x3"}
    return create_discipline("AnalyticDiscipline", expressions_dict=expressions)


@pytest.fixture
def parameter_space():  # type: (...) -> ParameterSpace
    """Return the parameter space on which to evaluate the discipline."""
    space = ParameterSpace()
    for name in ["x1", "x2", "x3"]:
        space.add_random_variable(name, "OTNormalDistribution")
    return space


class Ishigami1D(MDODiscipline):
    """A version of the Ishigami function indexed by a 1D variable."""

    def __init__(self):
        super(Ishigami1D, self).__init__()
        self.input_grammar.initialize_from_data_names(["x1", "x2", "x3"])
        self.output_grammar.initialize_from_data_names(["y"])

    def _run(self):
        x_1, x_2, x_3 = self.get_local_data_by_name(["x1", "x2", "x3"])
        time = linspace(0, 1, 100)
        output = sin(x_1) + 7 * sin(x_2) ** 2 + 0.1 * x_3 ** 4 * sin(x_1) * time
        self.store_local_data(y=output)


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
                "y1": [{"x1": array([1]), "x2": array([1, 1])}],
                "y2": [
                    {"x1": array([2]), "x2": array([2, 2])},
                    {"x1": array([3]), "x2": array([3, 3])},
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
def mock_sensitivity_analysis():  # type: (...) -> MockSensitivityAnalysis
    """Return an instance of MockSensitivityAnalysis."""
    return MockSensitivityAnalysis()


@pytest.fixture
def second_mock_sensitivity_analysis():  # type: (...) -> SecondMockSensitivityAnalysis
    """Return an instance of SecondMockSensitivityAnalysis."""
    return SecondMockSensitivityAnalysis()


def test_plot_bar(mock_sensitivity_analysis):
    """Check that a Barplot is created with plot_bar.

    Args:
        mock_sensitivity_analysis: The sensitivity indices.
    """
    plot = mock_sensitivity_analysis.plot_bar("y1", save=False, show=False)
    assert plot.title is None
    assert isinstance(plot, BarPlot)
    plot = mock_sensitivity_analysis.plot_bar("y1", title="foo", save=False, show=False)
    assert plot.title == "foo"


def test_plot_radar(mock_sensitivity_analysis):
    """Check that a RadarChart is created with plot_radar.

    Args:
        mock_sensitivity_analysis: The sensitivity indices.
    """
    plot = mock_sensitivity_analysis.plot_radar("y1", save=False, show=False)
    assert plot.title is None
    plot = mock_sensitivity_analysis.plot_radar(
        "y1", title="foo", save=False, show=False
    )
    assert plot.title == "foo"
    assert isinstance(plot, RadarChart)


def test_plot_comparison(discipline, parameter_space):
    """Check if the comparison of sensitivity indices works.

    Args:
        discipline: A discipline of interest.
        parameter_space: The parameter space related to this discipline.
    """
    spearman = CorrelationAnalysis(discipline, parameter_space, 10)
    spearman.compute_indices()
    pearson = CorrelationAnalysis(discipline, parameter_space, 10)
    pearson.main_method = pearson._PEARSON
    pearson.compute_indices()
    plot = pearson.plot_comparison(spearman, "y", save=False, show=False, title="foo")
    assert plot.title == "foo"
    assert isinstance(plot, BarPlot)
    plot = pearson.plot_comparison(
        spearman, "y", save=False, show=False, use_bar_plot=False
    )
    assert isinstance(plot, RadarChart)


def test_inputs_names(mock_sensitivity_analysis):
    """Check the value of the attribute input_names.

    Args:
        mock_sensitivity_analysis: The sensitivity indices.
    """
    assert mock_sensitivity_analysis.inputs_names == ["x1", "x2"]


@pytest.mark.parametrize("output", ["y1", ("y1", 0), ("y2", 0)])
def test_sort_parameters(mock_sensitivity_analysis, output):
    """Check if the parameters are well sorted.

    Args:
        mock_sensitivity_analysis: The sensitivity indices.
        output: The value used to sort the parameters.
    """
    parameters = mock_sensitivity_analysis.sort_parameters(output)
    assert parameters == ["x2", "x1"]


def test_convert_to_dataset(mock_sensitivity_analysis):
    """Check if the SensitivityIndices are well converted to Dataset.

    Args:
        mock_sensitivity_analysis: The sensitivity indices.
    """
    dataset = mock_sensitivity_analysis.export_to_dataset()
    assert isinstance(dataset, Dataset)
    assert dataset.row_names == ["x1(0)", "x2(0)", "x2(1)"]
    assert dataset.variables == ["y1", "y2"]
    assert dataset.groups == ["m1", "m2"]
    assert_array_equal(dataset.data["y1"], array([[1], [1], [1]]))
    assert_array_equal(dataset.data["y2"], array([[2, 3], [2, 3], [2, 3]]))


@pytest.fixture(scope="module")
def ishigami():  # type: (...) -> SobolAnalysis
    """Return the Sobol' analysis for the Ishigami function."""
    space = ParameterSpace()
    for variable in ["x1", "x2", "x3"]:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-pi, maximum=pi
        )

    sobol_analysis = SobolAnalysis(Ishigami1D(), space, 100)
    sobol_analysis.main_method = "total"
    sobol_analysis.compute_indices()
    return sobol_analysis


def test_plot_1d_field(ishigami):
    """Check if a 1D field is well plotted.

    Args:
        ishigami: The Sobol' analysis for the Ishigami function.
    """
    plot = ishigami.plot_field("y", save=False, show=False)
    assert isinstance(plot, Curves)


def test_plot_2d_field(ishigami):
    """Check if a 2D field is well plotted.

    Args:
        ishigami: The Sobol' analysis for the Ishigami function.
    """
    plot = ishigami.plot_field("y", save=False, show=False)
    assert isinstance(plot, Curves)

    properties = {"linestyle": "-", "color": ["b", "k", "r"]}
    times = linspace(0, 1, 10)
    mesh = array([[time1, time2] for time1 in times for time2 in times])
    plot = ishigami.plot_field(
        "y", save=False, show=False, properties=properties, mesh=mesh
    )
    assert isinstance(plot, Surfaces)
