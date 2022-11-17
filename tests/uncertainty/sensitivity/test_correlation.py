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

from pathlib import Path

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.api import create_discipline
from gemseo.uncertainty.sensitivity.correlation.analysis import CorrelationAnalysis
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


def test_correlation(correlation, tmp_wd):
    varnames = ["x1", "x2"]
    correlation.compute_indices()
    indices = correlation.indices
    assert set(indices.keys()) == set(correlation._ALGORITHMS.keys())
    pearson = indices["pearson"]
    assert set(pearson.keys()) == {"y1", "y2"}
    assert len(pearson["y1"]) == 1
    assert set(pearson["y1"][0].keys()) == {"x1", "x2"}
    assert correlation.spearman == indices["spearman"]
    assert pearson == correlation.pearson
    for name in varnames:
        assert len(pearson["y1"][0][name]) == 1
    for algo in correlation._ALGORITHMS:
        assert hasattr(correlation, algo)

    with pytest.raises(NotImplementedError):
        correlation.main_method = "foo"

    correlation.plot("y1", directory_path=tmp_wd)
    assert Path("correlation_analysis.png").exists()


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
