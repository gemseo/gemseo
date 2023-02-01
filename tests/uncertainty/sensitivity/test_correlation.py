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

import logging
import re
import sys

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.uncertainty.sensitivity.correlation.analysis import CorrelationAnalysis
from gemseo.utils.testing import image_comparison
from numpy.testing import assert_equal


@pytest.fixture
def correlation() -> CorrelationAnalysis:
    """A correlation analysis."""
    discipline = create_discipline(
        "AnalyticDiscipline", expressions={"y1": "x1+2*x2", "y2": "x1-2*x2"}
    )
    space = ParameterSpace()
    for name in ["x1", "x2"]:
        space.add_random_variable(name, "OTNormalDistribution")
    return CorrelationAnalysis([discipline], space, 100)


def test_correlation(correlation):
    correlation.compute_indices()
    indices = correlation.indices
    assert set(indices.keys()) == set(correlation._ALGORITHMS.keys())
    pearson = indices["pearson"]
    assert set(pearson.keys()) == {"y1", "y2"}
    assert len(pearson["y1"]) == 1
    assert set(pearson["y1"][0].keys()) == {"x1", "x2"}
    assert correlation.spearman == indices["spearman"]
    assert pearson == correlation.pearson
    for name in ["x1", "x2"]:
        assert len(pearson["y1"][0][name]) == 1
    for algo in correlation._ALGORITHMS:
        assert hasattr(correlation, algo)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "foo is not an sensitivity method; "
            "available ones are pcc, pearson, prcc, spearman, src, srrc, ssrrc."
        ),
    ):
        correlation.main_method = "foo"


def test_correlation_main_method(correlation, caplog):
    """Check a logged message when changing main method."""
    correlation.main_method = "prcc"
    _, log_level, log_message = caplog.record_tuples[0]
    assert log_level == logging.INFO
    assert log_message == ("Use prcc indices as main indices.")


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python 3.8 or greater")
@pytest.mark.parametrize("baseline_images", [["plot"]])
@pytest.mark.parametrize("output", ["y1", ("y1", 0)])
@image_comparison(None)
def test_correlation_plot(correlation, baseline_images, output):
    """Check CorrelationAnalysis.plot()."""
    correlation.compute_indices()
    correlation.plot(output, save=False, show=False)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python 3.8 or greater")
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
    indices = correlation.indices["spearman"]["y2"][0]
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
    correlation.save("foo.pkl")
    new_correlation = CorrelationAnalysis.load("foo.pkl")
    correlation.compute_indices()
    new_correlation.compute_indices()
    assert_equal(new_correlation.dataset.data, correlation.dataset.data)
    assert new_correlation.default_output == correlation.default_output
