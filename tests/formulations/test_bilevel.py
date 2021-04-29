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

from gemseo.formulations.bilevel import BiLevel
from gemseo.formulations.bilevel_test_helper import TestBilevelFormulationBase


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
