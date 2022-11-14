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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                        documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the OptimizationProblem-to-ProblemGeneric adapter."""
from __future__ import annotations

import pytest

p7core = pytest.importorskip("da.p7core", reason="pSeven is not available")

from numpy import array  # noqa: E402

from gemseo.algos.design_space import DesignSpace  # noqa: E402
from gemseo.algos.opt.core.pseven_problem_adapter import PSevenProblem  # noqa: E402
from gemseo.algos.opt_problem import OptimizationProblem  # noqa: E402
from gemseo.core.mdofunctions.mdo_function import (  # noqa: E402
    MDOFunction,
)
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction  # noqa: E402
from gemseo.core.mdofunctions.mdo_quadratic_function import (  # noqa: E402
    MDOQuadraticFunction,
)
from gemseo.problems.analytical.power_2 import Power2  # noqa: E402
from gemseo.problems.analytical.rosenbrock import Rosenbrock  # noqa: E402


@pytest.fixture(scope="module")
def rosenbrock() -> Rosenbrock:
    """The Rosenbrock problem."""
    return Rosenbrock()


@pytest.fixture(scope="module")
def power2() -> Power2:
    """The Power2 problem."""
    return Power2()


@pytest.fixture(scope="module")
def p7_rosenbrock(rosenbrock) -> PSevenProblem:
    """The pSeven adapter for the Rosenbrock problem."""
    return PSevenProblem(rosenbrock)


@pytest.fixture(scope="module")
def p7_power2(power2) -> PSevenProblem:
    """The pSeven adapter for the Power2 problem."""
    return PSevenProblem(power2)


@pytest.fixture
def problem(rosenbrock, power2, request) -> OptimizationProblem:
    """A Gemseo problem dispatched via request for generic parametrized tests."""
    if request.param == "Rosenbrock":
        return rosenbrock
    if request.param == "Power2":
        return power2
    raise ValueError(f"Invalid problem name: {request.param}")


@pytest.fixture
def p7_problem(p7_rosenbrock, p7_power2, request) -> PSevenProblem:
    """A pSeven problem dispatched via request for generic parametrized tests."""
    if request.param == "Rosenbrock":
        return p7_rosenbrock
    if request.param == "Power2":
        return p7_power2
    raise ValueError(f"Invalid problem name: {request.param}")


@pytest.mark.parametrize(
    ["p7_problem", "size"], [("Rosenbrock", 2), ("Power2", 3)], indirect=["p7_problem"]
)
def test_indirect(p7_problem, size):
    """Check the number of variables."""
    assert p7_problem.size_x() == size


@pytest.mark.parametrize(
    ["p7_problem", "names"],
    [("Rosenbrock", ["x!0", "x!1"]), ("Power2", ["x!0", "x!1", "x!2"])],
    indirect=["p7_problem"],
)
def test_variables_names(p7_problem, names):
    """Check the names of the variables."""
    assert p7_problem.variables_names() == names


@pytest.mark.parametrize(
    ["p7_problem", "index", "bounds"],
    [
        ("Rosenbrock", 0, (-2.0, 2.0)),
        ("Rosenbrock", 1, (-2.0, 2.0)),
        ("Power2", 0, (-1.0, 1.0)),
        ("Power2", 1, (-1.0, 1.0)),
        ("Power2", 2, (-1.0, 1.0)),
    ],
    indirect=["p7_problem"],
)
def test_variables_bounds(p7_problem, index, bounds):
    """Check the bounds of the variables."""
    assert all(p7_problem.variables_bounds(index) == bounds)


@pytest.mark.parametrize(
    ["p7_problem", "problem"],
    [("Rosenbrock", "Rosenbrock"), ("Power2", "Power2")],
    indirect=True,
)
def test_initial_guess(p7_problem, problem):
    """Check the initial guess."""
    assert (
        p7_problem.initial_guess() == problem.design_space.get_current_value().tolist()
    )


@pytest.mark.parametrize("p7_problem", ["Rosenbrock", "Power2"], indirect=True)
def test_size_f(p7_problem):
    """Check the number of objectives."""
    assert p7_problem.size_f() == 1


@pytest.mark.parametrize(
    ["p7_problem", "names"],
    [("Rosenbrock", ["rosen"]), ("Power2", ["pow2"])],
    indirect=["p7_problem"],
)
def test_objectives_names(p7_problem, names):
    """Check the objectives names."""
    assert p7_problem.objectives_names() == names


@pytest.mark.parametrize(
    ["p7_problem", "enabled", "sparse", "rows", "columns"],
    [("Rosenbrock", True, False, (), ()), ("Power2", True, False, (), ())],
    indirect=["p7_problem"],
)
def test_objectives_gradient(p7_problem, enabled, sparse, rows, columns):
    """Check the objectives gradients."""
    assert p7_problem.objectives_gradient() == (enabled, sparse, rows, columns)


@pytest.mark.parametrize(
    ["p7_problem", "number"],
    [("Rosenbrock", 0), ("Power2", 3)],
    indirect=["p7_problem"],
)
def test_size_c(p7_problem, number):
    """Check the number of constraints."""
    assert p7_problem.size_c() == number


@pytest.mark.parametrize(
    ["p7_problem", "names"],
    [("Rosenbrock", []), ("Power2", ["ineq1", "ineq2", "eq"])],
    indirect=["p7_problem"],
)
def test_constraints_names(p7_problem, names):
    """Check the constraints names."""
    assert p7_problem.constraints_names() == names


def test_constraints_bounds(p7_rosenbrock, p7_power2, power2):
    """Check the constraints bounds."""
    # Unconstrained problem
    lower_bounds, upper_bounds = p7_rosenbrock.constraints_bounds()
    assert lower_bounds == []
    assert upper_bounds == []

    # Constrained problem
    lower_bounds, upper_bounds = p7_power2.constraints_bounds()
    assert lower_bounds[2:] == [-power2.eq_tolerance]
    assert upper_bounds[2:] == [power2.eq_tolerance]
    assert upper_bounds[:2] == [power2.ineq_tolerance] * 2


@pytest.mark.parametrize(
    ["p7_problem", "enabled", "sparse", "rows", "columns"],
    [("Rosenbrock", False, False, (), ()), ("Power2", True, False, (), ())],
    indirect=["p7_problem"],
)
def test_constraints_gradient(p7_problem, enabled, sparse, rows, columns):
    """Check the constraints gradients."""
    assert p7_problem.constraints_gradient() == (enabled, sparse, rows, columns)


@pytest.mark.parametrize(
    ["p7_problem", "size"], [("Rosenbrock", 3), ("Power2", 16)], indirect=["p7_problem"]
)
def test_size_full(p7_problem, size):
    """Check the number of values computed."""
    assert p7_problem.size_full() == size


@pytest.mark.parametrize(
    ["p7_problem", "index", "key", "value"],
    [
        ("Rosenbrock", 0, "@GT/VariableType", "Continuous"),
        ("Rosenbrock", 1, "@GT/VariableType", "Continuous"),
        ("Rosenbrock", 2, "@GTOpt/LinearityType", "Generic"),
        ("Rosenbrock", 2, "@GTOpt/EvaluationCostType", None),
        ("Rosenbrock", 2, "@GTOpt/ExpensiveEvaluations", None),
        ("Power2", 0, "@GT/VariableType", "Continuous"),
        ("Power2", 1, "@GT/VariableType", "Continuous"),
        ("Power2", 2, "@GT/VariableType", "Continuous"),
        ("Power2", 3, "@GTOpt/LinearityType", "Generic"),
        ("Power2", 3, "@GTOpt/EvaluationCostType", None),
        ("Power2", 3, "@GTOpt/ExpensiveEvaluations", None),
        ("Power2", 4, "@GTOpt/LinearityType", "Generic"),
        ("Power2", 4, "@GTOpt/EvaluationCostType", None),
        ("Power2", 4, "@GTOpt/ExpensiveEvaluations", None),
        ("Power2", 5, "@GTOpt/LinearityType", "Generic"),
        ("Power2", 5, "@GTOpt/EvaluationCostType", None),
        ("Power2", 5, "@GTOpt/ExpensiveEvaluations", None),
        ("Power2", 6, "@GTOpt/LinearityType", "Generic"),
        ("Power2", 6, "@GTOpt/EvaluationCostType", None),
        ("Power2", 6, "@GTOpt/ExpensiveEvaluations", None),
    ],
    indirect=["p7_problem"],
)
def test_elements_hints(p7_problem, index, key, value):
    """Check the pSeven hints."""
    assert p7_problem.elements_hint(index, key) == value


@pytest.mark.parametrize(["size", "use_threading"], [[1, False], [2, False], [2, True]])
def test_unconstrained_evaluate(rosenbrock, size, use_threading):
    """Check the evaluation of an unconstrained problem functions."""
    queryx = array([[0.0, 0.0], [1.0, 1.0]])
    querymask = array([[1, 0, 0], [0, 1, 0]])
    ref_functions = [[1.0, None, None], [None, 0.0, 0.0]]
    ref_masks = [[True, False, False], [False, True, True]]
    assert PSevenProblem(rosenbrock, use_threading=use_threading).evaluate(
        queryx[:size], querymask[:size]
    ) == (ref_functions[:size], ref_masks[:size])


@pytest.mark.parametrize(["size", "use_threading"], [[1, False], [2, False], [2, True]])
def test_constrained_evaluate(power2, size, use_threading):
    """Check the evaluation of a constrained problem functions."""
    queryx = array([[1.0, 1.0, 1.0], [0.5 ** (1.0 / 3.0)] * 2 + [0.9 ** (1.0 / 3.0)]])
    querymask = array(
        [
            [1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        ]
    )
    ref_functions = [
        [
            3.0,
            -0.5,
            None,
            -0.1,
            2.0,
            2.0,
            2.0,
            -3.0,
            0.0,
            0.0,
            None,
            None,
            None,
            0.0,
            0.0,
            -3.0,
        ],
        [
            None,
            None,
            0.0,
            None,
            None,
            None,
            None,
            -3.0 / 4.0 ** (1.0 / 3.0),
            0.0,
            0.0,
            0.0,
            -3.0 / 4.0 ** (1.0 / 3.0),
            0.0,
            None,
            None,
            None,
        ],
    ]
    ref_masks = [
        [True, True, False, True] + [True] * 6 + [False] * 3 + [True] * 3,
        [False, False, True, False] + [False] * 3 + [True] * 6 + [False] * 3,
    ]
    functions_batch, output_masks_batch = PSevenProblem(
        power2, use_threading=use_threading
    ).evaluate(queryx[:size], querymask[:size])
    assert len(functions_batch) == size
    assert functions_batch[0] == pytest.approx(ref_functions[0])
    if size == 2:
        assert functions_batch[1] == pytest.approx(ref_functions[1])

    assert output_masks_batch == ref_masks[:size]


def test_multi_objectives():
    """Check the support of several objectives."""
    design_space = DesignSpace()
    design_space.add_variable("x", value=0.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: array([x[0], x[0] + 1, x[0] + 2]), "f")
    pseven_problem = PSevenProblem(problem)
    values, mask = pseven_problem.evaluate(array([[1.0]]), array([[True, True, False]]))
    assert values == [[1.0, 2.0, 3.0]]
    assert (mask == array([[True, True, True]])).all()


def test_multidimensional_constraint():
    """Check the support of a multidimensional constraint."""
    design_space = DesignSpace()
    design_space.add_variable("x", value=0.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "f")
    problem.add_ineq_constraint(
        MDOFunction(lambda x: array([x[0], x[0] + 1, x[0] + 2]), "g")
    )
    pseven_problem = PSevenProblem(problem)
    values, mask = pseven_problem.evaluate(
        array([[1.0]]), array([[True, False, True, False]])
    )
    assert values == [[1.0, 1.0, 2.0, 3.0]]
    assert (mask == array([[True, True, True, True]])).all()


def test_function_without_dimension():
    """Check the passing of a function without dimension."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "f")
    with pytest.raises(
        p7core.exceptions.BBPrepareException,
        match="Problem definition error: RuntimeError in black box prepare:"
        " The output dimension of function f is not available.",
    ):
        PSevenProblem(problem)


@pytest.mark.parametrize(
    ["gemseo_type", "pseven_type"], [("float", "Continuous"), ("integer", "Integer")]
)
def test_variables_types(gemseo_type, pseven_type):
    """Check the pSeven variables types."""
    design_space = DesignSpace()
    design_space.add_variable("x", var_type=gemseo_type, value=7)
    gemseo_problem = OptimizationProblem(design_space)
    gemseo_problem.objective = MDOFunction(lambda x: x, "f")
    pseven_problem = PSevenProblem(gemseo_problem)
    assert pseven_problem.elements_hint(0, "@GT/VariableType") == pseven_type


@pytest.mark.parametrize(
    ["evaluation_cost_type", "objective_type"],
    [
        ({"rosen": "Cheap"}, "Cheap"),
        ("Cheap", "Cheap"),
        ({"rosen": "Expensive"}, "Expensive"),
        ("Expensive", "Expensive"),
        ({"f": "Expensive"}, None),
    ],
)
def test_evaluation_cost_type_rosenbrock(
    rosenbrock, evaluation_cost_type, objective_type
):
    """Check the setting of an evaluation cost type on the Rosenbrock function."""
    problem = PSevenProblem(rosenbrock, evaluation_cost_type=evaluation_cost_type)
    assert problem.elements_hint(2, "@GTOpt/EvaluationCostType") == objective_type


@pytest.mark.parametrize(
    ["evaluation_cost_type"],
    [
        ({"pow2": "Cheap", "ineq1": "Cheap", "ineq2": "Expensive", "eq": "Expensive"},),
        ({"pow2": "Expensive", "ineq1": "Expensive", "ineq2": "Cheap", "eq": "Cheap"},),
        ({"pow2": "Cheap", "ineq1": "Expensive", "ineq2": "Cheap", "eq": "Expensive"},),
        ({"pow2": "Expensive", "ineq1": "Cheap", "ineq2": "Expensive", "eq": "Cheap"},),
    ],
)
def test_evaluation_cost_type_power2_individual(power2, evaluation_cost_type):
    """Check the setting of individual evaluation cost types on the Power2 functions."""
    problem = PSevenProblem(power2, evaluation_cost_type=evaluation_cost_type)
    assert (
        problem.elements_hint(3, "@GTOpt/EvaluationCostType")
        == evaluation_cost_type["pow2"]
    )
    assert (
        problem.elements_hint(4, "@GTOpt/EvaluationCostType")
        == evaluation_cost_type["ineq1"]
    )
    assert (
        problem.elements_hint(5, "@GTOpt/EvaluationCostType")
        == evaluation_cost_type["ineq2"]
    )
    assert (
        problem.elements_hint(6, "@GTOpt/EvaluationCostType")
        == evaluation_cost_type["eq"]
    )


@pytest.mark.parametrize(
    ["evaluation_cost_type"],
    [
        ("Cheap",),
        ("Expensive",),
    ],
)
def test_evaluation_cost_type_power2_common(power2, evaluation_cost_type):
    """Check the setting of a common evaluation cost type on the Power2 functions."""
    problem = PSevenProblem(power2, evaluation_cost_type=evaluation_cost_type)
    assert problem.elements_hint(3, "@GTOpt/EvaluationCostType") == evaluation_cost_type
    assert problem.elements_hint(4, "@GTOpt/EvaluationCostType") == evaluation_cost_type
    assert problem.elements_hint(5, "@GTOpt/EvaluationCostType") == evaluation_cost_type


@pytest.mark.parametrize(
    ["expensive_evaluations", "value"], [({"rosen": 7}, 7), ({"f": 7}, None)]
)
def test_expensive_evaluations(rosenbrock, expensive_evaluations, value):
    """Check the pSeven numbers of expensive evaluations."""
    problem = PSevenProblem(rosenbrock, expensive_evaluations=expensive_evaluations)
    assert problem.elements_hint(2, "@GTOpt/ExpensiveEvaluations") == value


@pytest.mark.parametrize(
    ["gemseo_type", "pseven_type"],
    [(MDOLinearFunction, "Linear"), (MDOQuadraticFunction, "Quadratic")],
)
def test_linearity_types(gemseo_type, pseven_type):
    """Check the pSeven linearity types."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    problem = OptimizationProblem(design_space)
    problem.objective = gemseo_type(array([[1.0]]), "f")
    linearity_type = PSevenProblem(problem).elements_hint(1, "@GTOpt/LinearityType")
    assert linearity_type == pseven_type
