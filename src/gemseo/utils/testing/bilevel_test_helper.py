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
    formulation_name: str = "BiLevel",
) -> Callable[[dict[str, Any]], MDOScenario]:
    """Create a function to generate a Sobieski Scenario.

    Args:
        formulation_name: The name of the formulation of the scenario.

    Returns:
        A function which generates a Sobieski scenario with specific options.
    """

    def func(**settings):
        """Create a Sobieski BiLevel scenario.

        Args:
             **settings: The settings of the system scenario.

        Returns:
            A Sobieski BiLevel Scenario.
        """
        sub_scenarios = create_sobieski_sub_scenarios()
        for scenario in sub_scenarios:
            scenario.set_algorithm(algo_name="SLSQP", max_iter=5)

        system = MDOScenario(
            [*sub_scenarios, SobieskiMission()],
            "y_4",
            SobieskiProblem().design_space.filter(["x_shared", "y_14"]),
            formulation_name=formulation_name,
            maximize_objective=True,
            **settings,
        )
        system.set_differentiation_method("finite_differences")
        return system

    return func


def create_sobieski_sub_scenarios() -> tuple[MDOScenario, MDOScenario, MDOScenario]:
    """Return the sub-scenarios of Sobieski's SuperSonic Business Jet."""
    design_space = SobieskiProblem().design_space
    propulsion = MDOScenario(
        [SobieskiPropulsion()],
        "y_34",
        design_space.filter("x_3", copy=True),
        name="PropulsionScenario",
        formulation_name="DisciplinaryOpt",
    )

    # Maximize L/D
    aerodynamics = MDOScenario(
        [SobieskiAerodynamics()],
        "y_24",
        design_space.filter("x_2", copy=True),
        formulation_name="DisciplinaryOpt",
        name="AerodynamicsScenario",
        maximize_objective=True,
    )

    # Maximize log(aircraft total weight / (aircraft total weight - fuel
    # weight))
    structure = MDOScenario(
        [SobieskiStructure()],
        "y_11",
        design_space.filter("x_1"),
        formulation_name="DisciplinaryOpt",
        name="StructureScenario",
        maximize_objective=True,
    )

    return structure, aerodynamics, propulsion
