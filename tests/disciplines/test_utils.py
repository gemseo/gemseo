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
from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.utils import get_all_inputs
from gemseo.disciplines.utils import get_all_outputs


@pytest.fixture(scope="module")
def disciplines_and_scenario() -> list[MDODiscipline]:
    """Disciplines with a scenario."""
    disciplines = [
        AnalyticDiscipline({"y1": "x1"}, name="f1"),
        AnalyticDiscipline({"y2": "x2"}, name="f2"),
    ]
    sub_disciplines = [
        AnalyticDiscipline({"ya": "xa"}, name="fa"),
        AnalyticDiscipline({"yb": "xb"}, name="fb"),
    ]
    design_space = DesignSpace()
    design_space.add_variable("xa")
    scenario = MDOScenario(sub_disciplines, "DisciplinaryOpt", "ya", design_space)
    return disciplines + [scenario]


@pytest.mark.parametrize(
    "skip_scenarios,expected", [(True, ["x1", "x2"]), (False, ["x1", "x2", "xa", "xb"])]
)
def test_get_all_inputs(disciplines_and_scenario, skip_scenarios, expected):
    """Check get_all_inputs."""
    all_inputs = get_all_inputs(disciplines_and_scenario, skip_scenarios=skip_scenarios)
    assert set(all_inputs) == set(expected)


@pytest.mark.parametrize(
    "skip_scenarios,expected", [(True, ["y1", "y2"]), (False, ["y1", "y2", "ya", "yb"])]
)
def test_get_all_outputs(disciplines_and_scenario, skip_scenarios, expected):
    """Check get_all_outputs."""
    all_inputs = get_all_outputs(
        disciplines_and_scenario, skip_scenarios=skip_scenarios
    )
    assert set(all_inputs) == set(expected)
