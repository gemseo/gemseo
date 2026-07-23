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
from numpy.testing import assert_almost_equal

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.uncertainty import create_sensitivity_analysis
from gemseo.uncertainty.distributions.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.openturns.sorm_settings import OT_SORM_Settings
from gemseo.uncertainty.reliability.scenario import ReliabilityScenario
from gemseo.uncertainty.sensitivity.form import FORMAnalysis
from gemseo.uncertainty.sensitivity.form import FORMAnalysisMethod


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """A differentiable discipline y = x1 + 2*x2."""
    return AnalyticDiscipline({"y": "x1 + 2*x2"}, name="my_function")


@pytest.fixture(scope="module")
def parameter_space() -> ParameterSpace:
    """The uncertain space of two standard normal variables."""
    space = ParameterSpace()
    space.add_random_variable("x1", OTNormalDistribution_Settings())
    space.add_random_variable("x2", OTNormalDistribution_Settings())
    return space


@pytest.fixture(scope="module")
def events(discipline, parameter_space):
    """The events of interest, indexed by their names."""
    analysis = FORMAnalysis()
    y = analysis.get_event_variables("y")
    return {"y_high": y > 1.0}


@pytest.fixture(scope="module")
def form(discipline, parameter_space, events) -> FORMAnalysis:
    """A FORM analysis."""
    analysis = FORMAnalysis()
    analysis.compute_samples([discipline], parameter_space, events)
    analysis.compute_indices()
    return analysis


def test_output_names(form) -> None:
    """Check the output names, which are the event names."""
    assert form.default_output_names == ["y_high"]


def test_main_method(form) -> None:
    """Check the default main method."""
    assert form.main_method == FORMAnalysisMethod.CLASSICAL


def test_main_indices(form) -> None:
    """Check the structure of the main sensitivity indices."""
    main_indices = form.main_indices
    assert set(main_indices) == {"y_high"}
    assert len(main_indices["y_high"]) == 1
    assert set(main_indices["y_high"][0]) == {"x1", "x2"}


@pytest.mark.parametrize("method", ["classical", "elliptical", "physical"])
def test_indices_methods(form, method) -> None:
    """Check that all the importance-factor types are populated."""
    indices = getattr(form.indices, method)
    assert set(indices) == {"y_high"}
    for input_name in ("x1", "x2"):
        assert indices["y_high"][0][input_name].size == 1


def test_dataset(form) -> None:
    """Check that the dataset stores the FORM evaluations and the result."""
    assert isinstance(form.dataset, IODataset)
    assert len(form.dataset) > 0
    assert set(form.dataset.input_names) == {"x1", "x2"}
    result = form.dataset.misc["execution_result"]["y_high"]
    assert isinstance(result.reliability_index, float)


def test_plot(form, tmp_wd) -> None:
    """Check that the plot works with an event name as output."""
    form.plot("y_high", save=True, show=False)


def test_sort_input_variables(form) -> None:
    """Check that the inputs can be sorted by influence on an event."""
    assert set(form.sort_input_variables("y_high")) == {"x1", "x2"}


def test_factory() -> None:
    """Check that the high-level API creates a FORMAnalysis."""
    assert isinstance(create_sensitivity_analysis("FORM"), FORMAnalysis)


def test_from_samples(form) -> None:
    """Check that a FORMAnalysis can be created from samples."""
    analysis = FORMAnalysis(form.dataset)
    analysis.compute_indices()
    assert analysis.indices.elliptical["y_high"] == form.indices.elliptical["y_high"]


def test_sorm(discipline, parameter_space, events) -> None:
    """Check that a SORM study can be used instead of FORM."""
    analysis = FORMAnalysis()
    analysis.compute_samples(
        [discipline], parameter_space, events, algo_settings=OT_SORM_Settings()
    )
    indices = analysis.compute_indices()
    assert set(indices.classical) == {"y_high"}


@pytest.mark.parametrize("use_database", [False, True])
def test_form_database(
    discipline, parameter_space, events, use_database, caplog
) -> None:
    """Check that a FORMAnalysis needs the database."""
    analysis = FORMAnalysis()
    algo_settings = OT_FORM_Settings(use_database=use_database)
    analysis.compute_samples(
        [discipline],
        parameter_space,
        events,
        algo_settings=algo_settings,
    )
    # The database has been used and converted into a Dataset.
    assert len(analysis.dataset) == 36
    # However, the settings have not been changed.
    assert algo_settings.use_database is use_database


def test_consistency_with_reliability_scenario(
    discipline, parameter_space, events, form
) -> None:
    """Cross-check the indices against a direct reliability study."""
    scenario = ReliabilityScenario([discipline], parameter_space)
    y = scenario.get_event_variables("y")
    scenario.add_event(y > 1.0, "y_high")
    scenario.execute(OT_FORM_Settings())
    result = scenario.event_name_to_reliability_result["y_high"]
    for input_name in ("x1", "x2"):
        assert_almost_equal(
            form.indices.classical["y_high"][0][input_name],
            result.importance_factors.classical_as_dict[input_name],
        )
