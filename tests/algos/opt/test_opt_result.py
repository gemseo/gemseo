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
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import unittest

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.utils.string_tools import MultiLineString


class TestOptResult(unittest.TestCase):
    def test_init_dict_repr(self):
        dct = {
            "x_0": [0],
            "x_opt": [1],
            "optimizer_name": "LBFGSB",
            "message": "msg",
            "f_opt": 1.1,
            OptimizationResult.HDF_CSTR_KEY + "cname": [0.0],
            "status": 1,
            "n_obj_call": 10,
            "n_grad_call": 10,
            "n_constr_call": 10,
            "is_feasible": True,
        }
        res = OptimizationResult.init_from_dict_repr(**dct)

        assert res.x_0 == dct["x_0"]
        assert res.optimizer_name == dct["optimizer_name"]
        assert res.message == dct["message"]
        assert res.f_opt == dct["f_opt"]
        assert (
            res.constraints_values["cname"]
            == dct[OptimizationResult.HDF_CSTR_KEY + "cname"]
        )
        assert res.status == dct["status"]
        assert res.n_obj_call == dct["n_obj_call"]
        assert res.n_grad_call == dct["n_grad_call"]
        assert res.n_constr_call == dct["n_constr_call"]
        assert res.is_feasible == dct["is_feasible"]

        self.assertRaises(ValueError, OptimizationResult.init_from_dict_repr, toto=4)


def test_str():
    """Check the string representation of an optimization result."""
    space = DesignSpace()
    space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    space.add_variable("z", size=2, l_b=0.0, u_b=1.0, value=0.5)
    disc = AnalyticDiscipline(
        expressions_dict={
            "y": "x",
            "eq_1": "x",
            "eq_2": "x",
            "ineq_p_1": "x",
            "ineq_p_2": "x",
            "ineq_n_1": "x",
            "ineq_n_2": "x",
        }
    )
    scenario = DOEScenario([disc], "DisciplinaryOpt", "y", space)
    scenario.add_constraint("eq_1", constraint_type="eq")
    scenario.add_constraint("eq_2", constraint_type="eq", value=0.25)
    scenario.add_constraint("ineq_p_1", constraint_type="ineq", positive=True)
    scenario.add_constraint("ineq_n_1", constraint_type="ineq", value=0.25)
    scenario.add_constraint(
        "ineq_p_2", constraint_type="ineq", positive=True, value=0.25
    )
    scenario.add_constraint("ineq_n_2", constraint_type="ineq")
    scenario.execute({"algo": "fullfact", "n_samples": 1})
    opt_result = scenario.optimization_result

    expected = MultiLineString()
    expected.add("Optimization result:")
    expected.add("Objective value = 0.5")
    expected.add("The result is not feasible.")
    expected.add("Status: None")
    expected.add("Optimizer message: None")
    expected.add("Number of calls to the objective function by the optimizer: 1")
    expected.add("Constraints values:")
    expected.indent()
    expected.add("-ineq_p_1 = -0.5")
    expected.add("-ineq_p_2 + 0.25 = -0.25")
    expected.add("eq_1 = 0.5")
    expected.add("eq_2 - 0.25 = 0.25")
    expected.add("ineq_n_1 - 0.25 = 0.25")
    expected.add("ineq_n_2 = 0.5")
    assert str(opt_result) == str(expected)
