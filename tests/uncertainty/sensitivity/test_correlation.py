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

from unittest import mock

import pytest

from gemseo import create_discipline
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.uncertainty.sensitivity.correlation import analysis
from gemseo.uncertainty.sensitivity.correlation.analysis import CorrelationAnalysis
from gemseo.utils.compatibility.openturns import IS_OT_LOWER_THAN_1_20
from gemseo.utils.testing.helpers import image_comparison


@pytest.fixture()
def correlation() -> CorrelationAnalysis:
    """A correlation analysis."""
    discipline = create_discipline(
        "AnalyticDiscipline", expressions={"y1": "x1+2*x2", "y2": "x1-2*x2"}
    )
    space = ParameterSpace()
    for name in ["x1", "x2"]:
        space.add_random_variable(name, "OTNormalDistribution")
    return CorrelationAnalysis([discipline], space, 100)


def test_compute_indices(correlation):
    """Check CorrelationAnalysis.compute_indices()."""
    correlation.compute_indices()
    indices = correlation.indices
    # Check the methods for which the indices have been computed.
    all_methods = set(correlation.Method)
    available_methods = set(indices.keys())
    if IS_OT_LOWER_THAN_1_20:
        assert available_methods == all_methods - {
            correlation.Method.KENDALL,
            correlation.Method.SSRC,
        }
    else:
        assert available_methods == all_methods

    # Check the names and sizes of the outputs.
    pearson = indices["Pearson"]
    output_names = {"y1", "y2"}
    assert set(pearson.keys()) == output_names
    for output_name in output_names:
        assert len(pearson[output_name]) == 1

    # Check the names and sizes of the inputs.
    input_names = {"x1", "x2"}
    assert set(pearson["y1"][0].keys()) == input_names
    for input_name in input_names:
        assert pearson["y1"][0][input_name].shape == (1,)

    # Check that the property ``method`` is ``indices[algo.lower()]``.
    for algo in correlation.Method:
        assert getattr(correlation, algo.lower()) == indices[algo]


@pytest.mark.parametrize("baseline_images", [["plot"]])
@pytest.mark.parametrize("output", ["y1", ("y1", 0)])
@image_comparison(None)
def test_correlation_plot(correlation, baseline_images, output):
    """Check CorrelationAnalysis.plot()."""
    correlation.compute_indices()
    correlation.plot(output, save=False, show=False)


@pytest.mark.parametrize("baseline_images", [(["plot_radar"])])
@image_comparison(None)
def test_correlation_plot_radar(correlation, baseline_images):
    """Check CorrelationAnalysis.plot_radar()."""
    correlation.compute_indices()
    correlation.plot_radar("y1", save=False, show=False)


def test_aggregate_sensitivity_indices(correlation):
    """Check _aggregate_sensitivity_indices()."""
    correlation.compute_indices()
    assert correlation.sort_parameters("y2") == ["x2", "x1"]
    indices = correlation.indices["Spearman"]["y2"][0]
    c1 = indices["x1"]
    c2 = indices["x2"]
    assert c2 < 0 < c1
    assert abs(c2) > abs(c1)


def test_correlation_outputs():
    expressions = {"y1": "x1+2*x2", "y2": "x1-2*x2"}
    varnames = ["x1", "x2"]
    discipline = create_discipline("AnalyticDiscipline", expressions=expressions)
    space = ParameterSpace()
    for variable in varnames:
        space.add_random_variable(variable, "OTUniformDistribution")

    correlation = CorrelationAnalysis([discipline], space, 100)
    correlation.compute_indices()
    assert {"y1", "y2"} == set(correlation.main_indices.keys())

    correlation = CorrelationAnalysis([discipline], space, 100)
    correlation.compute_indices("y1")
    assert {"y1"} == set(correlation.main_indices.keys())


def test_save_load(correlation, tmp_wd):
    """Check saving and loading a CorrelationAnalysis."""
    correlation.to_pickle("foo.pkl")
    new_correlation = CorrelationAnalysis.from_pickle("foo.pkl")
    correlation.compute_indices()
    new_correlation.compute_indices()
    assert new_correlation.dataset.equals(correlation.dataset)
    assert new_correlation.default_output == correlation.default_output


def test_mock_ot_version(correlation):
    """Check that KENDALL and SSRC are not available with openturns < 1.20."""
    indices = correlation.compute_indices()
    assert correlation.kendall
    assert correlation.ssrc
    assert correlation.Method.KENDALL in indices
    assert correlation.Method.SSRC in indices

    with mock.patch.object(analysis, "IS_OT_LOWER_THAN_1_20", new=True):
        indices = correlation.compute_indices()
        assert not correlation.kendall
        assert not correlation.ssrc
        assert correlation.Method.KENDALL not in indices
        assert correlation.Method.SSRC not in indices
