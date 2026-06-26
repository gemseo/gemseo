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
from openturns import SORMResult

from gemseo.uncertainty.reliability.openturns.optimizer import BaseOTOptimizer
from gemseo.uncertainty.reliability.openturns.sorm import OT_SORM
from gemseo.uncertainty.reliability.openturns.sorm_settings import OT_SORM_Settings
from gemseo.uncertainty.reliability.problem import ReliabilityProblem


@pytest.mark.parametrize("optimizer_type", [None, *BaseOTOptimizer.__subclasses__()])
@pytest.mark.parametrize("approximation", [None, *OT_SORM_Settings.Approximation])
@pytest.mark.parametrize(
    ("greater", "expected"), [(None, 0.25), (True, 0.25), (False, 0.75)]
)
def test_sorm(
    function, uncertain_space, approximation, greater, expected, optimizer_type
):
    """Test OT_SORM."""
    kwargs = {}
    if approximation is not None:
        kwargs["approximation"] = approximation
    if optimizer_type is not None:
        kwargs["optimizer"] = optimizer_type()
    if kwargs:
        kwargs = {"settings": OT_SORM_Settings(**kwargs)}

    sorm = OT_SORM()
    problem = ReliabilityProblem(uncertain_space)
    f = problem.get_event_variables(function)
    problem.add_event(
        f < 0.75 if greater is False else f > 0.75,
        event_name="a",
    )
    results = sorm.execute(problem, **kwargs)
    assert len(results) == 1
    assert results["a"].name == "a"
    probability = results["a"].probability
    reliability_index = results["a"].reliability_index
    raw_result = results["a"].raw_result
    assert probability == pytest.approx(expected, abs=1e-3)
    assert isinstance(raw_result, SORMResult)
    assert raw_result.getEventProbabilityBreitung() == probability
    assert raw_result.getHasoferReliabilityIndex() == reliability_index
