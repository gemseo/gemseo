# -*- coding: utf-8 -*-
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

"""Provide base test class stub for testing bilevel also for |g| plugins."""
from __future__ import division, unicode_literals

from copy import deepcopy
from typing import Callable, Dict

from gemseo.core.mdo_scenario import MDOScenario
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiProblem,
    SobieskiPropulsion,
    SobieskiStructure,
)

# TODO: remove when PEP 484 type hints will be used
FixtureFunc = Callable[[Dict[str, float]], MDOScenario]


def create_sobieski_bilevel_scenario():  # type: (...) -> FixtureFunc
    """Create a function to generate a Sobieski BiLevel Scenario.

    Returns:
        A function which generates a Sobieski BiLevel Scenario with specific options.
    """

    def func(**options):
        """Create a Sobieski BiLevel scenario.

        Args:
             **options: The options of the system scenario.

        Returns:
            A Sobieski BiLevel Scenario.
        """
        propulsion = SobieskiPropulsion()
        aerodynamics = SobieskiAerodynamics()
        struct = SobieskiStructure()
        mission = SobieskiMission()

        ds = SobieskiProblem().read_design_space()
        sc_prop = MDOScenario(
            disciplines=[propulsion],
            formulation="DisciplinaryOpt",
            objective_name="y_34",
            design_space=deepcopy(ds).filter("x_3"),
            name="PropulsionScenario",
        )

        # Maximize L/D
        sc_aero = MDOScenario(
            disciplines=[aerodynamics],
            formulation="DisciplinaryOpt",
            objective_name="y_24",
            design_space=deepcopy(ds).filter("x_2"),
            name="AerodynamicsScenario",
            maximize_objective=True,
        )

        # Maximize log(aircraft total weight / (aircraft total weight - fuel
        # weight))
        sc_str = MDOScenario(
            disciplines=[struct],
            formulation="DisciplinaryOpt",
            objective_name="y_11",
            design_space=deepcopy(ds).filter("x_1"),
            name="StructureScenario",
            maximize_objective=True,
        )

        sub_scenarios = [sc_str, sc_aero, sc_prop]
        sub_disciplines = sub_scenarios + [mission]
        for sc in sub_scenarios:
            sc.default_inputs = {"max_iter": 5, "algo": "SLSQP"}

        ds = SobieskiProblem().read_design_space()
        sc_system = MDOScenario(
            sub_disciplines,
            formulation="BiLevel",
            objective_name="y_4",
            design_space=ds.filter(["x_shared", "y_14"]),
            maximize_objective=True,
            **options
        )
        sc_system.set_differentiation_method("finite_differences", step=1e-6)
        return sc_system

    return func
