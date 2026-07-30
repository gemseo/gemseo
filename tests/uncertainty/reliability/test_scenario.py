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
from openturns import MultiFORMResult

from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.space.parameter import ParameterSpace
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.reliability.openturns.system_form_settings import (
    OT_SystemFORM_Settings,
)
from gemseo.uncertainty.reliability.scenario import ReliabilityScenario


def test_scenario():
    """Test ReliabilityScenario."""
    uncertain_space = ParameterSpace()
    uncertain_space.add_random_variable("u", OTUniformDistribution_Settings())
    uncertain_space.add_random_variable("v", OTUniformDistribution_Settings())

    discipline = AnalyticDiscipline({"y": "u", "z": "v"})

    scenario = ReliabilityScenario((discipline,), uncertain_space)
    y, z = scenario.get_event_variables("y", "z")
    scenario.add_event(y > 0.5, event_name="a")
    scenario.add_event((y > 0.6) & (z > 0.7), event_name="b")

    scenario.execute(OT_SystemFORM_Settings())
    result = scenario.event_name_to_reliability_result

    assert result["a"].probability == pytest.approx(0.5, abs=1e-4)
    assert isinstance(result["a"].raw_result, MultiFORMResult)

    assert result["b"].probability == pytest.approx(0.12, abs=1e-4)
    assert isinstance(result["b"].raw_result, MultiFORMResult)
