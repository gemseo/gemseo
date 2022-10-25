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
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.string_tools import MultiLineString


def test_from_dict():
    """Check the creation of an optimization result from a dictionary."""
    dct = {
        "x_0": [0],
        "x_opt": [1],
        "optimizer_name": "LBFGSB",
        "message": "msg",
        "f_opt": 1.1,
        "constr:cname": [0.0],
        "status": 1,
        "n_obj_call": 10,
        "n_grad_call": 10,
        "n_constr_call": 10,
        "is_feasible": True,
    }
    res = OptimizationResult.from_dict(dct)

    assert res.x_0 == dct["x_0"]
    assert res.optimizer_name == dct["optimizer_name"]
    assert res.message == dct["message"]
    assert res.f_opt == dct["f_opt"]
    assert res.constraints_values["cname"] == dct["constr:cname"]
    assert res.status == dct["status"]
    assert res.n_obj_call == dct["n_obj_call"]
    assert res.n_grad_call == dct["n_grad_call"]
    assert res.n_constr_call == dct["n_constr_call"]
    assert res.is_feasible == dct["is_feasible"]


@pytest.fixture(scope="module")
def optimization_result() -> OptimizationResult:
    """An optimization result."""
    space = DesignSpace()
    space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    space.add_variable("z", size=2, l_b=0.0, u_b=1.0, value=0.5)
    disc = AnalyticDiscipline(
        {
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
    return scenario.optimization_result


def test_repr(optimization_result):
    """Check the string representation of an optimization result."""
    expected = MultiLineString()
    expected.add("Optimization result:")
    expected.indent()
    expected.add("Design variables: [0.5]")
    expected.add("Objective function: 0.5")
    expected.add("Feasible solution: False")
    assert repr(optimization_result) == str(expected)


def test_str(optimization_result):
    """Check the string representation of an optimization result."""
    expected = MultiLineString()
    expected.add("Optimization result:")
    expected.indent()
    expected.add("Optimizer info:")
    expected.indent()
    expected.add("Status: None")
    expected.add("Message: None")
    expected.add("Number of calls to the objective function by the optimizer: 1")
    expected.dedent()
    expected.add("Solution:")
    expected.indent()
    expected.add("The solution is not feasible.")
    expected.add("Objective: 0.5")
    expected.add("Standardized constraints:")
    expected.indent()
    expected.add("-ineq_p_1 = -0.5")
    expected.add("-ineq_p_2 + 0.25 = -0.25")
    expected.add("eq_1 = 0.5")
    expected.add("eq_2 - 0.25 = 0.25")
    expected.add("ineq_n_1 - 0.25 = 0.25")
    expected.add("ineq_n_2 = 0.5")
    assert str(optimization_result) == str(expected)


def test_optimum_index(optimization_result):
    """Check the value of the optimum index of an optimization result."""
    assert optimization_result.optimum_index == 0


def test_default_optimum_index(caplog):
    """Check that the default value of the optimum index is None."""
    result = OptimizationResult()
    assert result.optimum_index is None


def test_initialize_optimum_index():
    """Check that the optimum index is correctly initialized."""
    result = OptimizationResult(optimum_index=1)
    assert result.optimum_index == 1
