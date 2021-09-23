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

# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

from copy import deepcopy

from gemseo.algos.design_space import DesignSpace
from gemseo.api import create_discipline, create_scenario
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.formulations.bilevel import BiLevel
from gemseo.formulations.bilevel_test_helper import TestBilevelFormulationBase
from gemseo.problems.aerostructure.aerostructure_design_space import (
    AerostructureDesignSpace,
)


class TestBilevelFormulation(TestBilevelFormulationBase):
    def test_execute(self):
        """"""
        scenario = self.build_bilevel(
            apply_cstr_tosub_scenarios=True, apply_cstr_to_system=False
        )

        for i in range(1, 4):
            scenario.add_constraint(["g_" + str(i)], "ineq")
        scenario.formulation.get_expected_workflow()

        for i in range(3):
            cstrs = scenario.disciplines[i].formulation.opt_problem.constraints
            assert len(cstrs) == 1
            assert cstrs[0].name == "g_" + str(i + 1)

        cstrs_sys = scenario.formulation.opt_problem.constraints
        assert len(cstrs_sys) == 0
        self.assertRaises(ValueError, scenario.add_constraint, ["toto"], "ineq")

    def test_get_sub_options_grammar(self):
        self.assertRaises(ValueError, BiLevel.get_sub_options_grammar)
        self.assertRaises(ValueError, BiLevel.get_default_sub_options_values)
        BiLevel.get_default_sub_options_values(mda_name="MDAJacobi")


def test_bilevel_aerostructure():
    """Test the Bi-level formulation on the aero-structure problem."""
    algo_options = {
        "xtol_rel": 1e-8,
        "xtol_abs": 1e-8,
        "ftol_rel": 1e-8,
        "ftol_abs": 1e-8,
        "ineq_tolerance": 1e-5,
        "eq_tolerance": 1e-3,
    }

    aero_formulas = {
        "drag": "0.1*((sweep/360)**2 + 200 + "
        + "thick_airfoils**2 - thick_airfoils - 4*displ)",
        "forces": "10*sweep + 0.2*thick_airfoils - 0.2*displ",
        "lift": "(sweep + 0.2*thick_airfoils - 2.*displ)/3000.",
    }
    aerodynamics = create_discipline(
        "AnalyticDiscipline", name="Aerodynamics", expressions_dict=aero_formulas
    )
    struc_formulas = {
        "mass": "4000*(sweep/360)**3 + 200000 + 100*thick_panels + 200.0*forces",
        "reserve_fact": "-3*sweep - 6*thick_panels + 0.1*forces + 55",
        "displ": "2*sweep + 3*thick_panels - 2.*forces",
    }
    structure = create_discipline(
        "AnalyticDiscipline", name="Structure", expressions_dict=struc_formulas
    )
    mission_formulas = {"range": "8e11*lift/(mass*drag)"}
    mission = create_discipline(
        "AnalyticDiscipline", name="Mission", expressions_dict=mission_formulas
    )
    sub_scenario_options = {
        "max_iter": 2,
        "algo": "NLOPT_SLSQP",
        "algo_options": algo_options,
    }
    design_space_ref = AerostructureDesignSpace()

    design_space_aero = deepcopy(design_space_ref).filter(["thick_airfoils"])
    aero_scenario = create_scenario(
        disciplines=[aerodynamics, mission],
        formulation="DisciplinaryOpt",
        objective_name="range",
        design_space=design_space_aero,
        maximize_objective=True,
    )
    aero_scenario.default_inputs = sub_scenario_options

    design_space_struct = deepcopy(design_space_ref).filter(["thick_panels"])
    struct_scenario = create_scenario(
        disciplines=[structure, mission],
        formulation="DisciplinaryOpt",
        objective_name="range",
        design_space=design_space_struct,
        maximize_objective=True,
    )
    struct_scenario.default_inputs = sub_scenario_options

    design_space_system = deepcopy(design_space_ref).filter(["sweep"])
    system_scenario = create_scenario(
        disciplines=[aero_scenario, struct_scenario, mission],
        formulation="BiLevel",
        objective_name="range",
        design_space=design_space_system,
        maximize_objective=True,
        mda_name="MDAJacobi",
        tolerance=1e-8,
    )
    system_scenario.add_constraint("reserve_fact", "ineq", value=0.5)
    system_scenario.add_constraint("lift", "eq", value=0.5)
    system_scenario.execute(
        {"algo": "NLOPT_COBYLA", "max_iter": 5, "algo_options": algo_options}
    )


def test_grammar_type():
    """Check that the grammar type is correctly used."""
    discipline = AnalyticDiscipline(
        expressions_dict={"y1": "z+x1+y2", "y2": "z+x2+2*y1"}
    )
    design_space = DesignSpace()
    design_space.add_variable("x1")
    design_space.add_variable("x2")
    design_space.add_variable("z")
    scn1 = MDOScenario(
        [discipline], "DisciplinaryOpt", "y1", design_space.filter(["x1"], copy=True)
    )
    scn2 = MDOScenario(
        [discipline], "DisciplinaryOpt", "y2", design_space.filter(["x2"], copy=True)
    )
    grammar_type = discipline.SIMPLE_GRAMMAR_TYPE
    formulation = BiLevel(
        [scn1, scn2],
        "y1",
        design_space.filter(["z"], copy=True),
        grammar_type=grammar_type,
    )
    assert formulation.chain.grammar_type == grammar_type
