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
from gemseo.uncertainty.reliability.openturns.faure_settings import OT_Faure_Settings
from gemseo.uncertainty.reliability.openturns.halton_settings import OT_Halton_Settings
from gemseo.uncertainty.reliability.openturns.haselgrove_settings import (
    OT_Haselgrove_Settings,
)
from gemseo.uncertainty.reliability.openturns.lhs_settings import OT_LHS_Settings
from gemseo.uncertainty.reliability.openturns.mc import OT_MC
from gemseo.uncertainty.reliability.openturns.mc_settings import OT_MC_Settings
from gemseo.uncertainty.reliability.openturns.reverse_halton_settings import (
    OT_Reverse_Halton_Settings,
)
from gemseo.uncertainty.reliability.problem import ReliabilityProblem


@pytest.mark.parametrize(
    "settings",
    [
        None,
        OT_MC_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
        ),
        OT_LHS_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
        ),
        OT_Faure_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
        ),
        OT_Halton_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
        ),
        OT_Reverse_Halton_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
        ),
        OT_Haselgrove_Settings(
            maximum_outer_sampling=900,
            maximum_standard_deviation=0.01,
            maximum_coefficient_of_variation=0.01,
        ),
    ],
)
@pytest.mark.parametrize(
    ("greater", "expected"), [(None, 0.25), (True, 0.25), (False, 0.75)]
)
def test_sampling(function, uncertain_space, settings, greater, expected):
    """Test OT_Sampling."""
    sampling = (
        OT_MC()
        if settings is None
        else RELIABILITY_ALGORITHM_FACTORY.create(settings.target_class_name)
    )
    kwargs = {}
    if settings is not None:
        kwargs["settings"] = settings

    problem = ReliabilityProblem(uncertain_space)
    f = problem.get_event_variables(function)
    problem.add_event(
        f < 0.75 if greater is False else f > 0.75,
        event_name="a",
    )
    results = sampling.execute(problem, **kwargs)
    assert len(results) == 1
    assert results["a"].name == "a"
    probability = results["a"].probability
    raw_result = results["a"].raw_result
    assert probability == pytest.approx(expected, abs=1e-1)
    assert isinstance(raw_result, ProbabilitySimulationResult)
    assert raw_result.getProbabilityEstimate() == probability
