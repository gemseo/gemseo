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
from __future__ import annotations

import pytest
from openturns import ProbabilitySimulationResult

from gemseo.uncertainty.reliability.factory import RELIABILITY_ALGORITHM_FACTORY
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.openturns.is_form_settings import (
    OT_IS_FORM_Settings,
)
from gemseo.uncertainty.reliability.openturns.is_na_settings import OT_IS_NA_Settings
from gemseo.uncertainty.reliability.openturns.is_spce_settings import (
    OT_IS_SPCE_Settings,
)
from gemseo.uncertainty.reliability.openturns.optimizer import OTCobyla
from gemseo.uncertainty.reliability.problem import ReliabilityProblem


@pytest.mark.parametrize(
    "settings",
    [
        None,
        OT_IS_NA_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
        ),
        OT_IS_NA_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
            quantile_level=0.2,
        ),
        OT_IS_SPCE_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
        ),
        OT_IS_SPCE_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
            quantile_level=0.2,
        ),
        OT_IS_FORM_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
            form_settings=OT_FORM_Settings(
                optimizer=OTCobyla(maximum_constraint_error=1e-3)
            ),
        ),
        OT_IS_FORM_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
            form_settings=OT_FORM_Settings(
                optimizer=OTCobyla(maximum_constraint_error=1e-3)
            ),
            quantile_level=0.2,
        ),
        OT_IS_FORM_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
            form_settings=OT_FORM_Settings(
                optimizer=OTCobyla(maximum_constraint_error=1e-3)
            ),
            control=True,
        ),
    ],
)
@pytest.mark.parametrize(
    ("greater", "expected"), [(None, 0.25), (True, 0.25), (False, 0.75)]
)
def test_importance_sampling(
    function, unbounded_uncertain_space, settings, greater, expected
):
    """Test for importance sampling algorithms."""
    class_name = "OT_IS_NA" if settings is None else settings.target_class_name
    algo = RELIABILITY_ALGORITHM_FACTORY.create(class_name)

    kwargs = {}
    if settings is not None:
        kwargs["settings"] = settings

    problem = ReliabilityProblem(unbounded_uncertain_space)
    f = problem.get_event_variables(function)
    problem.add_event(
        f < 0.75 if greater is False else f > 0.75,
        event_name="a",
    )
    results = algo.execute(problem, **kwargs)
    assert len(results) == 1
    assert results["a"].name == "a"
    probability = results["a"].probability
    raw_result = results["a"].raw_result
    assert probability == pytest.approx(expected, abs=1e-1)
    assert isinstance(raw_result, ProbabilitySimulationResult)
    assert raw_result.getProbabilityEstimate() == probability
