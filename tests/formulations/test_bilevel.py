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

from __future__ import absolute_import, division, unicode_literals

import unittest
from copy import deepcopy

from future import standard_library

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.formulations.bilevel import BiLevel
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiProblem,
    SobieskiPropulsion,
    SobieskiStructure,
)
from gemseo.third_party.junitxmlreq import link_to

standard_library.install_aliases()

configure_logger(SOFTWARE_NAME)


class Test_BilevelFormulation(unittest.TestCase):
    """ """

    @classmethod
    def setUpClass(cls):
        cls.propulsion = SobieskiPropulsion()
        cls.aerodynamics = SobieskiAerodynamics()
        cls.struct = SobieskiStructure()
        cls.mission = SobieskiMission()

    def build_scenarios(self):
        """ """
        ds = SobieskiProblem().read_design_space()
        sc_prop = MDOScenario(
            disciplines=[self.propulsion],
            formulation="DisciplinaryOpt",
            objective_name="y_34",
            design_space=deepcopy(ds).filter("x_3"),
            name="PropulsionScenario",
        )

        # Maximize L/D
        sc_aero = MDOScenario(
            disciplines=[self.aerodynamics],
            formulation="DisciplinaryOpt",
            objective_name="y_24",
            design_space=deepcopy(ds).filter("x_2"),
            name="AerodynamicsScenario",
            maximize_objective=True,
        )

        # Maximize log(aircraft total weight / (aircraft total weight - fuel
        # weight))
        sc_str = MDOScenario(
            disciplines=[self.struct],
            formulation="DisciplinaryOpt",
            objective_name="y_11",
            design_space=deepcopy(ds).filter("x_1"),
            name="StructureScenario",
            maximize_objective=True,
        )

        return sc_str, sc_aero, sc_prop

    def build_system_scenario(self, disciplines, **options):
        """

        :param disciplines:

        """
        # Maximize range (Breguet)
        ds = SobieskiProblem().read_design_space()
        # Add a coupling to DV but bielevel filters it
        sc_system = MDOScenario(
            disciplines,
            formulation="BiLevel",
            objective_name="y_4",
            design_space=ds.filter(["x_shared", "y_14"]),
            maximize_objective=True,
            **options
        )
        assert sc_system.formulation.opt_problem.design_space.variables_names == [
            "x_shared"
        ]
        sc_system.set_differentiation_method("finite_differences", step=1e-6)
        return sc_system

    def build_bilevel(self, **options):
        """ """
        sub_scenarios = self.build_scenarios()
        sub_disciplines = list(sub_scenarios) + [self.mission]
        for sc in sub_scenarios:
            sc.default_inputs = {"max_iter": 5, "algo": "SLSQP"}
        system_scenario = self.build_system_scenario(sub_disciplines, **options)

        return system_scenario

    @link_to("Req-MDO-1.5", "Req-MDO-1.6")
    def test_execute(self):
        """ """
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
