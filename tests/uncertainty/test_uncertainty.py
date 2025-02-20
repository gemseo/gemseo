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
from gemseo.uncertainty.statistics.parametric_statistics import ParametricStatistics

if TYPE_CHECKING:
    from gemseo.uncertainty.sensitivity.morris_analysis import MorrisAnalysis


@pytest.fixture(scope="module")
def analysis() -> MorrisAnalysis:
    discipline = AnalyticDiscipline(
        {"y": "sin(x1)+7*sin(x2)**2+0.1*a3**4*sin(x1)"}, name="Ishigami"
    )

    space = ParameterSpace()
    for variable in ["x1", "x2", "a3"]:
        space.add_random_variable(
            variable, "OTUniformDistribution", minimum=-pi, maximum=pi
        )

    # Create a sensitivity analysis computing samples.
    analysis = create_sensitivity_analysis("MorrisAnalysis")
    analysis.compute_samples([discipline], space, n_samples=0, n_replicates=5)

    return analysis


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"base_class_name": "OTDistribution"}, {"base_class_name": "SPDistribution"}],
)
def test_available_distributions(kwargs) -> None:
    """Check the function get_available_distributions."""
    distributions = get_available_distributions(**kwargs)
    base_class_name = kwargs.get("base_class_name")
    if base_class_name == "OTDistribution":
        assert "OTNormalDistribution" in distributions
        assert "SPNormalDistribution" not in distributions
    elif base_class_name == "SPDistribution":
        assert "OTNormalDistribution" not in distributions
        assert "SPNormalDistribution" in distributions
    else:
        assert "OTNormalDistribution" in distributions
        assert "SPNormalDistribution" in distributions


def test_create_distribution() -> None:
    distribution = create_distribution("OTNormalDistribution")
    assert distribution.mean == 0.0


def test_available_sensitivity_analysis() -> None:
    sensitivities = get_available_sensitivity_analyses()
    assert "MorrisAnalysis" in sensitivities


def test_create_sensitivity(analysis: MorrisAnalysis) -> None:
    """Check the function create_sensitivity()."""
    # Create a new sensitivity analysis from these samples.
    other_analysis = create_sensitivity_analysis("morris", samples=analysis.dataset)

    # Verify that the datasets are identical
    assert analysis.dataset is other_analysis.dataset


def test_create_statistics() -> None:
    n_samples = 100
    dataset = Dataset.from_array(default_rng().normal(size=(n_samples, 1)))
    stat = create_statistics(dataset)
    assert isinstance(stat, EmpiricalStatistics)
    stat = create_statistics(dataset, tested_distributions=["Normal", "Exponential"])
    assert isinstance(stat, ParametricStatistics)


def test_io_names(analysis: MorrisAnalysis):
    analysis.dataset.to_pickle("dataset.pkl")
    input_names = analysis._input_names
    output_names = analysis._output_names
    analysis_from_pickle = create_sensitivity_analysis("Sobol", "dataset.pkl")
    input_names_from_pickle = analysis_from_pickle._input_names
    output_names_from_pickle = analysis_from_pickle._output_names
    assert input_names == input_names_from_pickle
    assert output_names == output_names_from_pickle
