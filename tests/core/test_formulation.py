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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import division, unicode_literals

import math
import unittest

import numpy as np

from gemseo.algos.design_space import DesignSpace
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.formulation import MDOFormulation
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import SobieskiMission
from gemseo.utils.data_conversion import DataConversion


class TestMDOFormulation(unittest.TestCase):
    """"""

    def test_get_generator(self):
        """"""
        sm = SobieskiMission()
        ds = SobieskiProblem().read_design_space()
        f = MDOFormulation([sm], "y_4", ds)
        args = ["toto"]
        self.assertRaises(Exception, f._get_generator_with_inputs, *args)

        self.assertRaises(Exception, f._get_generator_from, *args)

    def test_cstrs(self):
        """"""
        sm = SobieskiMission()
        ds = SobieskiProblem().read_design_space()
        f = MDOFormulation([sm], "y_4", ds)
        prob = f.opt_problem
        assert not prob.has_constraints()
        f.add_constraint("y_4", constraint_name="toto")
        assert f.opt_problem.constraints[-1].name == "toto"

    #     def test_disciplines_runinputs(self):
    #         sm = SobieskiMission()
    #         rid = SobieskiProblem().get_default_inputs(sm.get_input_data_names())
    #         f = MDOFormulation('CustomFormulation', [sm], rid, "y_4", ["x_shared"])
    #         inpt = f.get_discipline_run_inputs(sm)
    #         for k in sm.get_input_data_names():
    #             assert(k in inpt)
    #         for k in inpt:
    #             assert(k in sm.get_input_data_names())
    #
    #         gt_rid = f.get_reference_input_data()
    #         for k in rid:
    #             assert k in gt_rid
    #
    #         self.assertRaises(Exception, MDOFormulation, [], rid,
    #                           "y_4", ["x_shared"])
    #         self.assertRaises(Exception, MDOFormulation, None, rid,
    #                           "y_4", ["x_shared"])
    #
    #         self.assertRaises(Exception, f.get_discipline_run_inputs, None)

    def test_jac_sign(self):
        """"""
        sm = SobieskiMission()
        f = MDOFormulation([sm], "y_4", ["x_shared"])

        g = MDOFunction(
            math.sin,
            name="G",
            f_type="ineq",
            jac=math.cos,
            expr="sin(x)",
            args=["x", "y"],
        )
        f.opt_problem.objective = g

        obj = f.opt_problem.objective
        self.assertAlmostEqual(obj(math.pi / 2), 1.0, 9)
        self.assertAlmostEqual(obj.jac(0.0), 1.0, 9)

    def test_get_x0(self):
        """"""
        _ = MDOFormulation(
            [SobieskiMission()], "y_4", SobieskiProblem().read_design_space()
        )

    def test_add_user_defined_constraint_error(self):
        """"""
        sm = SobieskiMission()
        f = MDOFormulation([sm], "y_4", ["x_shared"])
        self.assertRaises(Exception, f.add_constraint, "y_4", "None", "None")

    # =========================================================================
    #     def test_add_user_defined_constraint(self):
    #         sm = SobieskiMission()
    #         design_space = DesignSpace()
    #         design_space.add_variable("x_shared", 1)
    #
    #         f = MDOFormulation([sm], "y_4", design_space)
    #         _, add_to = f.add_constraint(
    #             'y_4', constraint_type="ineq", constraint_name="InEq")
    #         assert add_to
    # =========================================================================

    def test_get_values_array_from_dict(self):
        """"""
        a = DataConversion.dict_to_array({}, [])
        self.assertIsInstance(a, type(np.array([])))

    def test_get_mask_from_datanames(self):
        """"""
        a = MDOFormulation._get_mask_from_datanames(["y_1", "y_2", "y_3"], ["y_2"])[0][
            0
        ]
        self.assertEqual(a, 1)

    def test_x_mask(self):
        """"""
        sm = SobieskiMission()
        rid = SobieskiProblem().get_default_inputs(sm.get_input_data_names())
        dvs = ["x_shared", "y_14"]

        design_space = DesignSpace()
        design_space.add_variable("x_shared", 4)
        design_space.add_variable("y_14", 4)
        f = MDOFormulation([sm], "y_4", design_space)

        x = np.concatenate([rid[n] for n in dvs])
        f.mask_x(dvs, x, dvs)
        f.mask_x(dvs, x)
        f.mask_x_swap_order(dvs, x, dvs)

        f._get_x_mask_swap(dvs)
        self.assertRaises(Exception, f._get_x_mask_swap, dvs, ["toto"])

        self.assertRaises(ValueError, f.mask_x_swap_order, dvs + ["toto"], x)
        #         x_masked = f.mask_x_swap_order(dvs, x[0:3])

        f.mask_x_swap_order(
            ["x_shared"],
            x_vect=np.zeros(19),
            all_data_names=design_space.variables_names,
        )

        design_space.remove_variable("x_shared")
        design_space.add_variable("x_shared", 10)
        self.assertRaises(ValueError, f.mask_x, dvs, x)
        self.assertRaises(ValueError, f.mask_x_swap_order, dvs, x)

    def test_remove_sub_scenario_dv_from_ds(self):
        ds2 = DesignSpace()
        ds2.add_variable("y_14")
        ds2.add_variable("x")
        ds1 = DesignSpace()
        ds1.add_variable("x")
        sm = SobieskiMission()
        s1 = MDOScenario([sm], "IDF", "y_4", ds1)
        f2 = MDOFormulation([sm, s1], "y_4", ds2)
        assert "x" in f2.design_space.variables_names
        f2._remove_sub_scenario_dv_from_ds()
        assert "x" not in f2.design_space.variables_names

    def test_wrong_inputs(self):
        """"""
        dvs = ["x_shared", "y_14"]

        design_space = DesignSpace()
        for name in dvs:
            design_space.add_variable(name, 1)
        self.assertRaises(TypeError, MDOFormulation, [], "y_4", design_space)

    def test_get_obj(self):
        """"""
        sm = SobieskiMission()
        dvs = ["x_shared", "y_14"]

        design_space = DesignSpace()
        for name in dvs:
            design_space.add_variable(name, 1)

        f = MDOFormulation([sm], "Y5", design_space)
        self.assertRaises(Exception, lambda: f.get_objective())

    def test_get_x_mask(self):
        sm = SobieskiMission()
        dvs = ["x_shared", "y_14"]

        design_space = DesignSpace()
        for name in dvs:
            design_space.add_variable(name, 1)

        f = MDOFormulation([sm], "y_4", design_space)
        x = np.concatenate([np.ones(1)] * 2)
        xm = f.mask_x(dvs, x, dvs)
        f.unmask_x(dvs, xm)
        f.unmask_x_swap_order(dvs, xm)
        f.mask_x_swap_order(dvs, x, dvs)
        f.unmask_x_swap_order(dvs, x, dvs, x_full=x)
        self.assertTrue([True for i in range(len(dvs))], f._get_x_mask_swap(dvs, dvs))
        f._get_x_mask_swap(dvs)

        design_space.remove_variable("x_shared")
        design_space.add_variable("x_shared", 10)
        self.assertRaises(ValueError, f.unmask_x_swap_order, dvs, x)

    def test_get_expected_workflow(self):
        """"""
        sm = SobieskiMission()
        ds = SobieskiProblem().read_design_space()
        f = MDOFormulation([sm], "Y5", ds)
        self.assertRaises(Exception, f.get_expected_workflow)


def test_grammar_type():
    """Check that the grammar type is correctly stored."""
    discipline = AnalyticDiscipline(expressions_dict={"y": "x"})
    design_space = DesignSpace()
    design_space.add_variable("x")
    formulation = MDOFormulation(
        [discipline], "y", design_space, grammar_type="a_grammar_type"
    )
    assert formulation._grammar_type == "a_grammar_type"
