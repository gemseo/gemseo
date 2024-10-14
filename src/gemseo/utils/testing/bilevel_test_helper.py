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
"""Provide base test class stub for testing BiLevel also for |g| plugins."""

from __future__ import annotations

from typing import Any
from typing import Callable

from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario


def create_sobieski_bilevel_scenario(
    scenario_formulation: str = "BiLevel",
) -> Callable[[dict[str, Any]], MDOScenario]:
    """Create a function to generate a Sobieski Scenario.

    Args:
        scenario_formulation: The name of the formulation of the scenario.

    Returns:
        A function which generates a Sobieski scenario with specific options.
    """

    def func(**options):
        """Create a Sobieski BiLevel scenario.

        Args:
             **options: The options of the system scenario.

        Returns:
            A Sobieski BiLevel Scenario.
        """
        sub_scenarios = create_sobieski_sub_scenarios()
        for scenario in sub_scenarios:
            scenario.default_input_data = {"max_iter": 5, "algo": "SLSQP"}

        system = MDOScenario(
            [*sub_scenarios, SobieskiMission()],
            scenario_formulation,
            "y_4",
            SobieskiProblem().design_space.filter(["x_shared", "y_14"]),
            maximize_objective=True,
            **options,
        )
        system.set_differentiation_method("finite_differences")
        return system

    return func


def create_sobieski_sub_scenarios() -> tuple[MDOScenario, MDOScenario, MDOScenario]:
    """Return the sub-scenarios of Sobieski's SuperSonic Business Jet."""
    design_space = SobieskiProblem().design_space
    propulsion = MDOScenario(
        [SobieskiPropulsion()],
        "DisciplinaryOpt",
        "y_34",
        design_space.filter("x_3", copy=True),
        "PropulsionScenario",
    )

    # Maximize L/D
    aerodynamics = MDOScenario(
        [SobieskiAerodynamics()],
        "DisciplinaryOpt",
        "y_24",
        design_space.filter("x_2", copy=True),
        "AerodynamicsScenario",
        maximize_objective=True,
    )

    # Maximize log(aircraft total weight / (aircraft total weight - fuel
    # weight))
    structure = MDOScenario(
        [SobieskiStructure()],
        "DisciplinaryOpt",
        "y_11",
        design_space.filter("x_1"),
        "StructureScenario",
        maximize_objective=True,
    )

    return structure, aerodynamics, propulsion
