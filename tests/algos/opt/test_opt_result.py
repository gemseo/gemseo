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
from numpy import array

from gemseo import execute_algo
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.repr_html import REPR_HTML_WRAPPER


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
    assert res.constraint_values["cname"] == dct["constr:cname"]
    assert res.status == dct["status"]
    assert res.n_obj_call == dct["n_obj_call"]
    assert res.n_grad_call == dct["n_grad_call"]
    assert res.n_constr_call == dct["n_constr_call"]
    assert res.is_feasible == dct["is_feasible"]


@pytest.fixture(scope="module")
def optimization_result() -> OptimizationResult:
    """An optimization result."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("z", size=2, l_b=0.0, u_b=1.0, value=0.5)
    disc = AnalyticDiscipline({
        "y": "x",
        "eq_1": "x",
        "eq_2": "x",
        "ineq_p_1": "x",
        "ineq_p_2": "x",
        "ineq_n_1": "x",
        "ineq_n_2": "x",
    })
    scenario = DOEScenario([disc], "DisciplinaryOpt", "y", design_space)
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


def test_optimization_result(optimization_result):
    """Check optimization_result."""
    assert optimization_result == OptimizationResult(
        x_0=array([0.5]),
        x_0_as_dict={"x": array([0.5])},
        x_opt=array([0.5]),
        x_opt_as_dict={"x": array([0.5])},
        f_opt=0.5,
        objective_name="y",
        optimizer_name="fullfact",
        n_obj_call=1,
        optimum_index=0,
        constraint_values={
            "-ineq_p_1": -0.5,
            "[ineq_n_1-0.25]": 0.25,
            "-[ineq_p_2-0.25]": -0.25,
            "ineq_n_2": 0.5,
            "eq_1": 0.5,
            "[eq_2-0.25]": 0.25,
        },
        constraints_grad={
            "-ineq_p_1": None,
            "[ineq_n_1-0.25]": None,
            "-[ineq_p_2-0.25]": None,
            "ineq_n_2": None,
            "eq_1": None,
            "[eq_2-0.25]": None,
        },
    )


def test_repr(optimization_result):
    """Check OptimizationResult.__repr__."""
    assert (
        repr(optimization_result)
        == """Optimization result:
   Design variables: [0.5]
   Objective function: 0.5
   Feasible solution: False"""
    )


def test_repr_html(optimization_result):
    """Check OptimizationResult._repr_html_."""
    assert optimization_result._repr_html_() == REPR_HTML_WRAPPER.format(
        "Optimization result:<br/>"
        "<ul>"
        "<li>Design variables: [0.5]</li>"
        "<li>Objective function: 0.5</li>"
        "<li>Feasible solution: False</li>"
        "</ul>"
    )


def test_str(optimization_result):
    """Check the string representation of an optimization result."""
    expected = """Optimization result:
   Optimizer info:
      Status: None
      Message: None
      Number of calls to the objective function by the optimizer: 1
   Solution:
      The solution is not feasible.
      Objective: 0.5
      Standardized constraints:
         -[ineq_p_2-0.25] = -0.25
         -ineq_p_1 = -0.5
         [eq_2-0.25] = 0.25
         [ineq_n_1-0.25] = 0.25
         eq_1 = 0.5
         ineq_n_2 = 0.5"""
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


def test_from_optimization_problem_empy_database():
    """Check from_optimization_problem with empty database."""
    problem = OptimizationProblem(DesignSpace())
    result = OptimizationResult.from_optimization_problem(problem)
    assert result == OptimizationResult(n_obj_call=0)


@pytest.mark.parametrize(
    ("value", "is_feasible", "sign", "constraint"),
    [(-1.0, False, "+", 1.1), (1.0, True, "-", -0.9)],
)
@pytest.mark.parametrize("use_standardized_objective", [True, False])
@pytest.mark.parametrize("maximize", [True, False])
def test_from_optimization_problem(
    value, is_feasible, sign, constraint, use_standardized_objective, maximize
):
    """Check from_optimization_problem with empty database."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "f")
    problem.add_constraint(
        MDOFunction(lambda x: x, "g"), value, MDOFunction.ConstraintType.INEQ
    )
    if maximize:
        problem.change_objective_sign()
    problem.use_standardized_objective = use_standardized_objective
    execute_algo(problem, "CustomDOE", "doe", samples=array([[0.1]]))
    result = OptimizationResult.from_optimization_problem(problem)
    f_opt = -0.1 if maximize and use_standardized_objective else 0.1

    objective_name = "-f" if maximize and use_standardized_objective else "f"
    assert result == OptimizationResult(
        x_0=array([0.1]),
        x_0_as_dict={"x": array([0.1])},
        x_opt=array([0.1]),
        x_opt_as_dict={"x": array([0.1])},
        f_opt=f_opt,
        objective_name=objective_name,
        n_obj_call=1,
        optimum_index=0,
        is_feasible=is_feasible,
        constraint_values={f"[g{sign}1.0]": array([constraint])},
        constraints_grad={f"[g{sign}1.0]": None},
    )
