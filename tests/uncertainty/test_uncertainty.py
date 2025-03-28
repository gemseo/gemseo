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

from typing import TYPE_CHECKING

import pytest
from numpy import pi
from numpy.random import default_rng

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.datasets.dataset import Dataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.uncertainty import create_distribution
from gemseo.uncertainty import create_sensitivity_analysis
from gemseo.uncertainty import create_statistics
from gemseo.uncertainty import get_available_distributions
from gemseo.uncertainty import get_available_sensitivity_analyses
from gemseo.uncertainty.statistics.empirical_statistics import EmpiricalStatistics
from gemseo.uncertainty.statistics.ot_parametric_statistics import (
    OTParametricStatistics,
)
from gemseo.uncertainty.statistics.sp_parametric_statistics import (
    SPParametricStatistics,
)

if TYPE_CHECKING:
    from gemseo.uncertainty.sensitivity.morris_analysis import MorrisAnalysis


@pytest.fixture(scope="module")
def analysis() -> MorrisAnalysis:
    """A Morris analysis."""
    discipline = AnalyticDiscipline(
        {"y": "sin(x1)+7*sin(x2)**2+0.1*a3**4*sin(x1)"}, name="Ishigami"
    )

    space = ParameterSpace()
    for variable in ["x1", "x2", "a3"]:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-pi, maximum=pi
        )

    analysis = create_sensitivity_analysis("MorrisAnalysis")
    analysis.compute_samples([discipline], space, n_samples=0, n_replicates=5)
    return analysis


@pytest.fixture(scope="module")
def samples() -> Dataset():
    """100 realizations of a standard Gaussian variable."""
    return Dataset.from_array(default_rng().normal(size=(100, 1)))


@pytest.mark.parametrize(
    ("kwargs", "ot_in", "sp_in"),
    [
        ({}, True, True),
        ({"base_class_name": "OTDistribution"}, True, False),
        ({"base_class_name": "SPDistribution"}, False, True),
    ],
)
def test_available_distributions(kwargs, ot_in, sp_in) -> None:
    """Verify that get_available_distributions can filter the distributions."""
    distributions = get_available_distributions(**kwargs)
    assert ("OTNormalDistribution" in distributions) is ot_in
    assert ("SPNormalDistribution" in distributions) is sp_in


def test_create_distribution() -> None:
    """Verify that create_distribution can create a distribution."""
    distribution = create_distribution("OTNormalDistribution")
    assert distribution.mean == 0.0


def test_available_sensitivity_analysis() -> None:
    """Verify that get_available_sensitivity_analyses returns sensitivity analyses."""
    sensitivities = get_available_sensitivity_analyses()
    assert "MorrisAnalysis" in sensitivities


def test_create_sensitivity(analysis: MorrisAnalysis) -> None:
    """Check the function create_sensitivity()."""
    # Create a new sensitivity analysis from these samples.
    other_analysis = create_sensitivity_analysis("morris", samples=analysis.dataset)

    # Verify that the datasets are identical
    assert analysis.dataset is other_analysis.dataset


@pytest.mark.parametrize(
    ("kwargs", "type_"),
    [
        ({}, EmpiricalStatistics),
        ({"tested_distributions": ["Normal", "Exponential"]}, OTParametricStatistics),
        ({"tested_distributions": ["norm", "expon"]}, SPParametricStatistics),
    ],
)
def test_create_statistics(samples, kwargs, type_) -> None:
    """Verify the type of statistics class in function of tested_distributions."""
    stat = create_statistics(samples, **kwargs)
    assert isinstance(stat, type_)


def test_io_names(tmp_wd, analysis):
    """Verify that pickling preserves the input and output names."""
    analysis.dataset.to_pickle("dataset.pkl")
    input_names = analysis._input_names
    output_names = analysis._output_names
    analysis_from_pickle = create_sensitivity_analysis("Sobol", "dataset.pkl")
    input_names_from_pickle = analysis_from_pickle._input_names
    output_names_from_pickle = analysis_from_pickle._output_names
    assert input_names == input_names_from_pickle
    assert output_names == output_names_from_pickle
