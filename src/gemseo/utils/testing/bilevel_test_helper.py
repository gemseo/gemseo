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

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.problems.mdo.aerostructure.aerostructure_design_space import (
    AerostructureDesignSpace,
)
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.testing.disciplines_creator import create_disciplines_from_desc


def create_sobieski_bilevel_scenario(
    formulation_name: str = "BiLevel",
) -> Callable[[dict[str, Any]], MDOScenario]:
    """Create a function to generate a BiLevel Sobieski Scenario.

    Args:
        formulation_name: The name of the formulation of the scenario.

    Returns:
        A function which generates a BiLevel Sobieski scenario with specific options.
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
    """Creates the sub-scenarios for the Sobieski's SSBJ problem.

    Returns:
        The sub-scenarios of Sobieski's SuperSonic Business Jet.
    """
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


def create_sobieski_bilevel_bcd_scenario() -> Callable[[dict[str, Any]], MDOScenario]:
    """Create a function to generate a BiLevel BCD Sobieski Scenario.

    Returns:
        A function which generates a BiLevel BCD Sobieski scenario
        with specific options.
    """

    def func(**settings):
        """Create a Sobieski BiLevel scenario.

        Args:
             **settings: The settings of the system scenario.

        Returns:
            A Sobieski BiLevel Scenario.
        """
        propulsion = SobieskiPropulsion()
        aerodynamics = SobieskiAerodynamics()
        struct = SobieskiStructure()
        mission = SobieskiMission()
        sub_disciplines = [struct, propulsion, aerodynamics, mission]

        ds = SobieskiProblem().design_space

        def create_block(design_var, name="MDOScenario"):
            scenario = MDOScenario(
                sub_disciplines,
                "y_4",
                ds.filter([design_var], copy=True),
                formulation_name="MDF",
                main_mda_name="MDAGaussSeidel",
                maximize_objective=True,
                name=name,
            )
            scenario.set_algorithm(max_iter=50, algo_name="SLSQP")
            scenario.formulation.optimization_problem.objective *= 0.001
            return scenario

        sc_prop = create_block("x_3", "PropulsionScenario")
        sc_prop.add_constraint("g_3", constraint_type="ineq")

        sc_aero = create_block("x_2", "AerodynamicsScenario")
        sc_aero.add_constraint("g_2", constraint_type="ineq")

        sc_str = create_block("x_1", "StructureScenario")
        sc_str.add_constraint("g_1", constraint_type="ineq")

        # Gather the sub-scenarios and mission for objective computation
        sub_scenarios = [sc_aero, sc_str, sc_prop, sub_disciplines[-1]]

        sc_system = MDOScenario(
            sub_scenarios,
            "y_4",
            ds.filter(["x_shared"], copy=True),
            formulation_name="BiLevelBCD",
            maximize_objective=True,
            bcd_mda_settings={"tolerance": 1e-5, "max_mda_iter": 10},
            **settings,
        )
        sc_system.formulation.optimization_problem.objective *= 0.001
        sc_system.set_differentiation_method("finite_differences")
        return sc_system

    return func


def create_dummy_bilevel_scenario(formulation_name: str) -> MDOScenario:
    """Create a dummy BiLevel scenario.

    It has to be noted that there is no strongly coupled discipline in this example.
    It implies that MDA1 will not be created. Yet, MDA2 will be created,
    as it is built with all the sub-disciplines passed to the BiLevel formulation.

    Args:
        formulation_name: The name of the BiLevel formulation to be used.

    Returns:
        A dummy BiLevel MDOScenario.
    """
    disc_expressions = {
        "disc_1": (["x_1"], ["a"]),
        "disc_2": (["a", "x_2"], ["b"]),
        "disc_3": (["x", "x_3", "b"], ["obj"]),
    }
    discipline_1, discipline_2, discipline_3 = create_disciplines_from_desc(
        disc_expressions
    )

    system_design_space = create_design_space()
    system_design_space.add_variable("x_3")

    sub_design_space_1 = create_design_space()
    sub_design_space_1.add_variable("x_1")
    sub_scenario_1 = create_scenario(
        [discipline_1, discipline_3]
        + ([discipline_2] if formulation_name == "BiLevelBCD" else []),
        "obj",
        sub_design_space_1,
        formulation_name="MDF",
    )

    sub_design_space_2 = create_design_space()
    sub_design_space_2.add_variable("x_2")
    sub_scenario_2 = create_scenario(
        [discipline_2, discipline_3]
        + ([discipline_1] if formulation_name == "BiLevelBCD" else []),
        "obj",
        sub_design_space_2,
        formulation_name="MDF",
    )

    return create_scenario(
        [sub_scenario_1, sub_scenario_2],
        "obj",
        design_space=system_design_space,
        formulation_name=formulation_name,
    )


def create_aerostructure_scenario(formulation_name: str):
    """Create an Aerostructure scenario.

    Args:
        formulation_name: The name of the BiLevel formulation to be used.

    Returns:
        An executed Aerostructure scenario.
    """
    algo_settings = {
        "xtol_rel": 1e-8,
        "xtol_abs": 1e-8,
        "ftol_rel": 1e-8,
        "ftol_abs": 1e-8,
        "ineq_tolerance": 1e-5,
        "eq_tolerance": 1e-3,
    }

    aero_formulas = {
        "drag": "0.1*((sweep/360)**2 + 200 + thick_airfoils**2 - thick_airfoils - "
        "4*displ)",
        "forces": "10*sweep + 0.2*thick_airfoils - 0.2*displ",
        "lift": "(sweep + 0.2*thick_airfoils - 2.*displ)/3000.",
    }
    aerodynamics = create_discipline(
        "AnalyticDiscipline", name="Aerodynamics", expressions=aero_formulas
    )
    struc_formulas = {
        "mass": "4000*(sweep/360)**3 + 200000 + 100*thick_panels + 200.0*forces",
        "reserve_fact": "-3*sweep - 6*thick_panels + 0.1*forces + 55",
        "displ": "2*sweep + 3*thick_panels - 2.*forces",
    }
    structure = create_discipline(
        "AnalyticDiscipline", name="Structure", expressions=struc_formulas
    )
    mission_formulas = {"range": "8e11*lift/(mass*drag)"}
    mission = create_discipline(
        "AnalyticDiscipline", name="Mission", expressions=mission_formulas
    )
    sub_scenario_options = {
        "max_iter": 2,
        "algo_name": "NLOPT_SLSQP",
        **algo_settings,
    }
    design_space_ref = AerostructureDesignSpace()

    design_space_aero = design_space_ref.filter(["thick_airfoils"], copy=True)

    aero_scenario = create_scenario(
        [aerodynamics, mission]
        + ([structure] if formulation_name == "BiLevelBCD" else []),
        "range",
        design_space=design_space_aero,
        formulation_name=(
            "DisciplinaryOpt" if formulation_name == "BiLevel" else "MDF"
        ),
        maximize_objective=True,
    )

    aero_scenario.set_algorithm(**sub_scenario_options)

    design_space_struct = design_space_ref.filter(["thick_panels"], copy=True)
    struct_scenario = create_scenario(
        [structure, mission]
        + ([aerodynamics] if formulation_name == "BiLevelBCD" else []),
        "range",
        design_space=design_space_struct,
        formulation_name=(
            "DisciplinaryOpt" if formulation_name == "BiLevel" else "MDF"
        ),
        maximize_objective=True,
    )
    struct_scenario.set_algorithm(**sub_scenario_options)

    design_space_system = design_space_ref.filter(["sweep"], copy=True)
    system_scenario = create_scenario(
        [aero_scenario, struct_scenario, mission],
        "range",
        design_space=design_space_system,
        formulation_name=formulation_name,
        maximize_objective=True,
        main_mda_name="MDAJacobi",
        main_mda_settings={"tolerance": 1e-8},
    )

    system_scenario.add_constraint("reserve_fact", constraint_type="ineq", value=0.5)
    system_scenario.add_constraint("lift", value=0.5)
    system_scenario.execute(algo_name="NLOPT_COBYLA", max_iter=5, **algo_settings)

    return system_scenario
