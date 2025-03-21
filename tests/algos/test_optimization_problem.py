# Copyright 2022 Airbus SAS
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
#        :author: Francois Gallard, Gabriel Max De Mendonça Abrantes
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import operator
import re
from copy import deepcopy
from functools import partial
from math import sqrt
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from numpy import allclose
from numpy import array
from numpy import array_equal
from numpy import cos
from numpy import inf
from numpy import ndarray
from numpy import ones
from numpy import sin
from numpy import zeros
from numpy.testing import assert_equal
from pandas import MultiIndex
from pandas.testing import assert_frame_equal
from scipy.linalg import norm
from scipy.optimize import rosen
from scipy.optimize import rosen_der

from gemseo import create_design_space
from gemseo import create_scenario
from gemseo import execute_algo
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.custom_doe import CustomDOE
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.doe.pydoe.pydoe import PyDOELibrary
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FunctionIsNan
from gemseo.algos.stop_criteria import MaxIterReachedException
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.problems.multiobjective_optimization.binh_korn import BinhKorn
from gemseo.problems.optimization.power_2 import Power2
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.repr_html import REPR_HTML_WRAPPER

DIRNAME = Path(__file__).parent
FAIL_HDF = DIRNAME / "fail2.hdf5"


@pytest.fixture(scope="module")
def problem_executed_twice() -> OptimizationProblem:
    """A problem executed twice."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "obj")
    problem.add_observable(MDOFunction(lambda x: x, "obs"))
    problem.add_constraint(MDOFunction(lambda x: x, "cstr", f_type="ineq"))

    execute_algo(
        problem, algo_name="CustomDOE", algo_type="doe", samples=array([[0.0]])
    )
    problem.evaluation_counter.current = 0
    execute_algo(
        problem, algo_name="CustomDOE", algo_type="doe", samples=array([[0.5]])
    )
    return problem


@pytest.fixture
def pow2_problem() -> OptimizationProblem:
    design_space = DesignSpace()
    design_space.add_variable("x", 3, lower_bound=-1.0, upper_bound=1.0)
    x_0 = np.ones(3)
    design_space.set_current_value(x_0)

    problem = OptimizationProblem(design_space)
    power2 = Power2()
    problem.objective = MDOFunction(
        power2.pow2,
        name="pow2",
        f_type="obj",
        jac=power2.pow2_jac,
        expr="x[0]**2+x[1]**2+x[2]**2",
        input_names=["x"],
    )
    return problem


def test_init() -> None:
    design_space = DesignSpace()
    OptimizationProblem(design_space)


def test_checks() -> None:
    n = 3
    design_space = DesignSpace()
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(rosen, name="rosen", f_type="obj", jac=rosen_der)

    with pytest.raises(ValueError):
        problem.design_space.set_current_value(np.zeros(n))
    with pytest.raises(ValueError):
        problem.design_space.set_upper_bound("x", np.ones(n))
    with pytest.raises(ValueError):
        problem.design_space.set_lower_bound("x", -np.ones(n))

    with pytest.raises(ValueError):
        problem.check()

    design_space.add_variable("x")
    problem.check()


def test_callback() -> None:
    """Test the execution of a callback."""
    n = 3
    design_space = DesignSpace()
    design_space.add_variable("x", n, lower_bound=-1.0, upper_bound=1.0)
    design_space.set_current_value(np.zeros(n))
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(rosen, name="rosen", f_type="obj", jac=rosen_der)
    problem.check()

    call_me = mock.Mock()
    problem.add_listener(call_me)
    problem.preprocess_functions()
    problem.check()

    problem.objective.evaluate(problem.design_space.get_current_value())
    call_me.assert_called_once()


def test_add_constraints(pow2_problem) -> None:
    problem = pow2_problem
    ineq1 = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq1",
        f_type=MDOFunction.ConstraintType.INEQ,
        jac=Power2.ineq_constraint1_jac,
        expr="0.5 -x[0] ** 3",
        input_names=["x"],
    )
    problem.add_constraint(ineq1, value=-1)
    assert len(tuple(problem.constraints.get_inequality_constraints())) == 1
    assert len(tuple(problem.constraints.get_equality_constraints())) == 0

    problem.add_constraint(ineq1, value=-1)
    problem.add_constraint(ineq1, value=-1)

    assert len(problem.constraints) == 3
    assert problem.constraints

    ineq2 = MDOFunction(Power2.ineq_constraint1, name="ineq2")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Constraint type must be provided, "
            "either when defining the function or when adding it to the problem."
        ),
    ):
        problem.add_constraint(ineq2)

    problem.add_constraint(ineq1, positive=True)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The function type 'obj' is not one of those authorized (eq, ineq)."
        ),
    ):
        problem.constraints.append(problem.objective)


def test_linear_problem_type_switch() -> None:
    n = 3
    design_space = DesignSpace()
    design_space.add_variable("x", n, lower_bound=-1.0, upper_bound=1.0)
    design_space.set_current_value(np.zeros(n))
    problem = OptimizationProblem(design_space)
    f = MDOFunction(Power2.ineq_constraint1, name="f")
    problem_c = deepcopy(problem)
    problem.objective = f
    assert not problem.is_linear
    problem_c.add_constraint(f, constraint_type=MDOFunction.ConstraintType.INEQ)
    assert not problem_c.is_linear


def test_getmsg_ineq_constraints(pow2_problem) -> None:
    expected = []
    problem = pow2_problem

    ineq_std = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_std",
        f_type="ineq",
        expr="cstr + cst",
        input_names=["x"],
    )
    problem.add_constraint(ineq_std)
    expected.append("ineq_std(x): cstr + cst <= 0.0")

    ineq_lo_posval = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_lo_posval",
        f_type="ineq",
        expr="cstr + cst",
        input_names=["x"],
    )
    problem.add_constraint(ineq_lo_posval, value=1.0)
    expected.append("ineq_lo_posval(x): cstr + cst <= 1.0")

    ineq_lo_negval = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_lo_negval",
        f_type="ineq",
        expr="cstr + cst",
        input_names=["x"],
    )
    problem.add_constraint(ineq_lo_negval, value=-1.0)
    expected.append("ineq_lo_negval(x): cstr + cst <= -1.0")

    ineq_up_negval = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_up_negval",
        f_type="ineq",
        expr="cstr + cst",
        input_names=["x"],
    )
    problem.add_constraint(ineq_up_negval, value=-1.0, positive=True)
    expected.append("ineq_up_negval(x): cstr + cst >= -1.0")

    ineq_up_posval = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_up_posval",
        f_type="ineq",
        expr="cstr + cst",
        input_names=["x"],
    )
    problem.add_constraint(ineq_up_posval, value=1.0, positive=True)
    expected.append("ineq_up_posval(x): cstr + cst >= 1.0")

    linear_constraint = MDOLinearFunction(array([1, 2]), "lin1", f_type="ineq")
    problem.add_constraint(linear_constraint)
    expected.append("lin1(x[0], x[1]): x[0] + 2.00e+00*x[1] <= 0.0")

    linear_constraint = MDOLinearFunction(array([1, 2]), "lin2", f_type="ineq")
    problem.add_constraint(linear_constraint, positive=True, value=-1.0)
    expected.append("lin2(x[0], x[1]): x[0] + 2.00e+00*x[1] >= -1.0")

    msg = str(problem)
    for elem in expected:
        assert elem in msg


def test_getmsg_eq_constraints(pow2_problem) -> None:
    expected = []
    problem = pow2_problem

    eq_std = MDOFunction(
        Power2.ineq_constraint1,
        name="eq_std",
        f_type="eq",
        expr="cstr + cst",
        input_names=["x"],
    )
    problem.add_constraint(eq_std)
    expected.append("eq_std(x): cstr + cst == 0.0")

    eq_posval = MDOFunction(
        Power2.ineq_constraint1,
        name="eq_posval",
        f_type="eq",
        expr="cstr + cst",
        input_names=["x"],
    )
    problem.add_constraint(eq_posval, value=1.0)
    expected.append("eq_posval(x): cstr + cst == 1.0")
    eq_negval = MDOFunction(
        Power2.ineq_constraint1,
        name="eq_negval",
        f_type="eq",
        expr="cstr + cst",
        input_names=["x"],
    )
    problem.add_constraint(eq_negval, value=-1.0)
    expected.append("eq_negval(x): cstr + cst == -1.0")

    msg = str(problem)
    for elem in expected:
        assert elem in msg


def test_get_dimension(pow2_problem) -> None:
    problem = pow2_problem
    problem.u_bounds = None
    problem.l_bounds = None
    dim = 3
    assert problem.design_space.dimension == dim
    problem.u_bounds = np.ones(3)
    assert problem.design_space.dimension == dim
    problem.l_bounds = -np.ones(3)
    assert problem.design_space.dimension == dim


def test_constraints_dim(pow2_problem) -> None:
    problem = pow2_problem
    ineq1 = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq1",
        f_type=MDOFunction.ConstraintType.INEQ,
        jac=Power2.ineq_constraint1_jac,
        expr="0.5 -x[0] ** 3",
        input_names=["x"],
    )
    problem.add_constraint(ineq1, value=-1)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The function output dimension is not available yet, "
            "please call function ineq1(x): 0.5 -x[0] ** 3 <= -1 once."
        ),
    ):
        problem.constraints.get_dimension(
            problem.constraints.get_inequality_constraints()
        )
    assert len(tuple(problem.constraints.get_equality_constraints())) == 0
    original_constraint = next(problem.constraints.get_originals())
    assert id(next(problem.constraints.get_inequality_constraints())) == id(
        original_constraint
    )
    problem.preprocess_functions()
    assert id(next(problem.constraints.get_inequality_constraints())) != id(
        original_constraint
    )


def test_check() -> None:
    # Objective is missing!
    design_space = DesignSpace()
    design_space.add_variable("x", 3, lower_bound=-1.0, upper_bound=1.0)
    design_space.set_current_value(np.array([1.0, 1.0, 1.0]))
    problem = OptimizationProblem(design_space)
    with pytest.raises(ValueError):
        problem.check()


def test_missing_constjac(pow2_problem) -> None:
    problem = pow2_problem
    ineq1 = MDOFunction(
        sum,
        name="sum",
        f_type=MDOFunction.ConstraintType.INEQ,
        expr="sum(x)",
        input_names=["x"],
    )
    problem.add_constraint(ineq1, value=-1)
    problem.preprocess_functions()
    output_functions, jacobian_functions = problem.get_functions(jacobian_names=())
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "The function computing the Jacobian of [sum+1] is not implemented."
        ),
    ):
        problem.evaluate_functions(
            ones(3),
            output_functions=output_functions,
            jacobian_functions=jacobian_functions,
        )


def _test_check_bounds(pow2_problem) -> None:
    dim = 3
    problem = pow2_problem
    problem.x_0 = np.ones(dim)

    problem.design_space.set_upper_bound("x", np.ones(dim))
    problem.design_space.set_lower_bound("x", np.array(dim * [-1]))
    with pytest.raises(TypeError):
        problem.check()

    problem.design_space.set_lower_bound("x", np.ones(dim))
    problem.design_space.set_upper_bound("x", np.array(dim * [-1]))
    with pytest.raises(TypeError):
        problem.check()

    problem.design_space.set_lower_bound("x", -np.ones(dim + 1))
    problem.design_space.set_upper_bound("x", np.ones(dim))
    with pytest.raises(ValueError):
        problem.check()

    problem.design_space.set_lower_bound("x", -np.ones(dim))
    problem.design_space.set_upper_bound("x", np.ones(dim))
    x_0 = np.ones(dim + 1)
    problem.design_space.set_current_value(x_0)
    with pytest.raises(ValueError):
        problem.check()

    problem.design_space.set_lower_bound("x", np.ones(dim) * 2)
    problem.design_space.set_upper_bound("x", np.ones(dim))
    x_0 = np.ones(dim)
    problem.design_space.set_current_value(x_0)
    with pytest.raises(ValueError):
        problem.check()


def test_invalid_differentiation_method(pow2_problem) -> None:
    """Check the error raised when using an invalid differentiation method."""
    pow2_problem.differentiation_method = "foo"
    with pytest.raises(ImportError, match=r"The class foo is not available"):
        pow2_problem.preprocess_functions()


def test_get_dv_names() -> None:
    problem = Power2()
    OptimizationLibraryFactory().execute(problem, algo_name="SLSQP")
    assert problem.design_space.variable_names == ["x"]


def test_get_best_infeasible_point() -> None:
    problem = Power2()
    problem.preprocess_functions()
    x_0 = problem.design_space.normalize_vect(zeros(3))
    f_val = problem.objective.evaluate(x_0)
    x_opt, f_opt, is_opt_feasible, opt_fd = (
        problem.history._OptimizationHistory__get_best_infeasible_point()
    )
    assert is_opt_feasible
    assert (x_opt == zeros(3)).all()
    assert f_opt == f_val
    assert "pow2" in opt_fd

    problem = Power2()
    problem.preprocess_functions()
    x_1 = problem.design_space.normalize_vect(array([-1.0, 0.0, 0.0]))
    problem.evaluate_functions(x_1)
    x_2 = problem.design_space.normalize_vect(array([0.0, -1.0, 0.0]))
    problem.evaluate_functions(x_2)
    x_opt, f_opt, is_opt_feasible, opt_fd = (
        problem.history._OptimizationHistory__get_best_infeasible_point()
    )
    assert not is_opt_feasible
    assert x_opt is not None
    assert f_opt is not None
    assert len(opt_fd) > 0
    f_last, x_last, is_feas, _, _ = problem.history.last_point
    assert allclose(x_last, array([0.0, -1.0, 0.0]))
    assert f_last == problem.objective.evaluate(x_2)
    assert is_feas == problem.constraints.is_point_feasible(
        problem.evaluate_functions(x_2)[0]
    )


def test_feasible_optimum_points() -> None:
    problem = Power2()
    with pytest.raises(ValueError):
        problem.optimum  # noqa:B018
    with pytest.raises(ValueError):
        problem.history.last_point  # noqa: B018

    OptimizationLibraryFactory().execute(
        problem, algo_name="SLSQP", eq_tolerance=1e-6, ineq_tolerance=1e-6
    )
    feasible_x = problem.history.feasible_points[0]
    assert len(feasible_x) >= 2
    min_value, solution, is_feasible, _, _ = problem.optimum
    assert (solution == feasible_x[-1]).all()
    assert allclose(min_value, 2.192090802, 9)
    assert allclose(solution[0], 0.79370053, 8)
    assert allclose(solution[1], 0.79370053, 8)
    assert allclose(solution[2], 0.96548938, 8)
    assert is_feasible


def test_nan() -> None:
    problem = Power2()
    problem.preprocess_functions()

    with pytest.raises(DesvarIsNan):
        problem.objective.evaluate(array([1.0, float("nan")]))

    with pytest.raises(DesvarIsNan):
        problem.objective.jac(array([1.0, float("nan")]))

    problem = Power2()
    problem.objective.jac = lambda x: array([float("nan")] * 3)
    problem.preprocess_functions()
    with pytest.raises(FunctionIsNan):
        problem.objective.jac(array([0.1, 0.2, 0.3]))


def test_preprocess_functions() -> None:
    """Test the pre-processing of a problem functions."""
    problem = Power2()
    obs1 = MDOFunction(norm, "design Euclidean norm")
    problem.add_observable(obs1)
    obs2 = MDOFunction(partial(norm, inf), "design infinity norm")
    problem.add_observable(obs2)

    # Store the initial functions identities
    obj_id = id(problem.objective)
    cstr_id = {id(cstr) for cstr in problem.constraints}
    obs_id = {id(obs) for obs in problem.observables}

    problem.preprocess_functions(is_function_input_normalized=False, round_ints=False)

    # Check that the non-preprocessed functions are the original ones
    assert id(problem.objective.original) == obj_id
    assert {id(cstr) for cstr in problem.constraints.get_originals()} == cstr_id
    assert {id(obs) for obs in problem.observables.get_originals()} == obs_id

    # Check that the current problem functions are NOT the original ones
    assert id(problem.objective) != obj_id
    assert {id(cstr) for cstr in problem.constraints}.isdisjoint(cstr_id)
    assert {id(obs) for obs in problem.observables}.isdisjoint(obs_id)

    nonproc_constraints = {repr(cstr) for cstr in problem.constraints.get_originals()}
    constraints = {repr(cstr) for cstr in problem.constraints}
    assert nonproc_constraints == constraints


def test_normalize_linear_function() -> None:
    """Test the normalization of linear functions."""
    design_space = DesignSpace()
    lower_bounds = array([-5.0, -7.0])
    upper_bounds = array([11.0, 13.0])
    x_0 = 0.2 * lower_bounds + 0.8 * upper_bounds
    design_space.add_variable(
        "x", 2, lower_bound=lower_bounds, upper_bound=upper_bounds, value=x_0
    )
    objective = MDOLinearFunction(
        array([[2.0, 0.0], [0.0, 3.0]]), "affine", "obj", "x", array([5.0, 7.0])
    )
    low_bnd_value = objective.evaluate(lower_bounds)
    upp_bnd_value = objective.evaluate(upper_bounds)
    initial_value = objective.evaluate(x_0)
    problem = OptimizationProblem(design_space)
    problem.objective = objective
    problem.preprocess_functions(use_database=False, round_ints=False)
    assert allclose(problem.objective.evaluate(zeros(2)), low_bnd_value)
    assert allclose(problem.objective.evaluate(ones(2)), upp_bnd_value)
    assert allclose(problem.objective.evaluate(0.8 * ones(2)), initial_value)


def test_export_hdf(tmp_wd) -> None:
    file_path = Path("power2.h5")
    problem = Power2()
    OptimizationLibraryFactory().execute(problem, algo_name="SLSQP")
    problem.to_hdf(file_path, append=True)  # Shall still work now

    def check_pb(imp_pb) -> None:
        assert file_path.exists()
        assert str(imp_pb) == str(problem)
        assert str(imp_pb.solution) == str(problem.solution)
        get_dimension = imp_pb.constraints.get_dimension
        assert get_dimension(imp_pb.constraints.get_equality_constraints()) == 1
        assert get_dimension(imp_pb.constraints.get_inequality_constraints()) == 2
        assert not imp_pb.is_linear
        assert imp_pb.differentiation_method == "user"

    problem.to_hdf(file_path)

    new_pbm = OptimizationProblem(problem.design_space, database=problem.database)
    assert new_pbm.database == problem.database

    imp_pb = OptimizationProblem.from_hdf(file_path)
    check_pb(imp_pb)

    problem.to_hdf(file_path, append=True)
    imp_pb = OptimizationProblem.from_hdf(file_path)
    check_pb(imp_pb)
    val = imp_pb.objective.evaluate(imp_pb.database.get_x_vect(2))
    assert isinstance(val, float)
    jac = imp_pb.objective.jac(imp_pb.database.get_x_vect(1))
    assert isinstance(jac, ndarray)
    with pytest.raises(ValueError):
        imp_pb.objective.evaluate(array([1.1254]))


def test_evaluate_functions() -> None:
    """Evaluate the functions of the Power2 problem."""
    problem = Power2()
    output_functions, jacobian_functions = problem.get_functions(
        jacobian_names=(),
        evaluate_objective=False,
    )
    func, grad = problem.evaluate_functions(
        design_vector=array([1.0, 0.5, 0.2]),
        design_vector_is_normalized=False,
        output_functions=output_functions,
        jacobian_functions=jacobian_functions,
    )
    assert "pow2" not in func
    assert "pow2" not in grad
    assert func["ineq1"] == pytest.approx(array([-0.5]))
    assert func["ineq2"] == pytest.approx(array([0.375]))
    assert func["eq"] == pytest.approx(array([0.892]))
    assert grad["ineq1"] == pytest.approx(array([-3.0, 0.0, 0.0]))
    assert grad["ineq2"] == pytest.approx(array([0.0, -0.75, 0.0]))
    assert grad["eq"] == pytest.approx(array([0.0, 0.0, -0.12]))


def test_evaluate_functions_no_gradient() -> None:
    """Evaluate the functions of the Power2 problem without computing the gradients."""
    problem = Power2()
    output_functions, jacobian_functions = problem.get_functions(
        no_db_no_norm=True, evaluate_objective=False
    )
    func, grad = problem.evaluate_functions(
        design_vector_is_normalized=False,
        jacobian_functions=jacobian_functions or None,
        output_functions=output_functions or None,
    )
    assert "pow2" not in func
    assert "pow2" not in grad
    assert func["ineq1"] == pytest.approx(array([-0.5]))
    assert func["ineq2"] == pytest.approx(array([-0.5]))
    assert func["eq"] == pytest.approx(array([-0.1]))


def test_evaluate_functions_only_gradients() -> None:
    """Evaluate the gradients of the Power2 problem without evaluating the functions."""
    problem = Power2()
    output_functions, jacobian_functions = problem.get_functions(
        no_db_no_norm=True,
        evaluate_objective=False,
        constraint_names=None,
        jacobian_names=["ineq1", "ineq2", "eq"],
    )
    func, grad = problem.evaluate_functions(
        design_vector_is_normalized=False,
        output_functions=output_functions or None,
        jacobian_functions=jacobian_functions or None,
    )
    assert not func
    assert grad.keys() == {"ineq1", "ineq2", "eq"}
    assert grad["ineq1"] == pytest.approx(array([-3, 0, 0]))
    assert grad["ineq2"] == pytest.approx(array([0, -3, 0]))
    assert grad["eq"] == pytest.approx(array([0, 0, -3]))


@pytest.mark.parametrize("no_db_no_norm", [True, False])
def test_evaluate_functions_w_observables(pow2_problem, no_db_no_norm) -> None:
    """Test the evaluation of the functions of a problem with observables."""
    problem = pow2_problem
    design_norm = "design norm"
    observable = MDOFunction(norm, design_norm)
    problem.add_observable(observable)
    problem.preprocess_functions()
    output_functions, jacobian_functions = problem.get_functions(
        no_db_no_norm=no_db_no_norm
    )
    out = problem.evaluate_functions(
        design_vector=array([1.0, 1.0, 1.0]),
        design_vector_is_normalized=False,
        output_functions=output_functions or None,
        jacobian_functions=jacobian_functions or None,
    )
    assert out[0]["pow2"] == pytest.approx(3.0)
    assert out[0]["design norm"] == pytest.approx(sqrt(3.0))


def test_evaluate_functions_non_preprocessed(constrained_problem) -> None:
    """Check the evaluation of non-preprocessed functions."""
    output_functions, jacobian_functions = constrained_problem.get_functions(
        no_db_no_norm=True, observable_names=None
    )
    values, jacobians = constrained_problem.evaluate_functions(
        design_vector_is_normalized=False,
        output_functions=output_functions or None,
        jacobian_functions=jacobian_functions or None,
    )
    assert set(values.keys()) == {"f", "g", "h"}
    assert values["f"] == pytest.approx(2.0)
    assert values["g"] == pytest.approx(array([1.0]))
    assert values["h"] == pytest.approx(array([1.0, 1.0]))
    assert jacobians == {}


@pytest.mark.parametrize(
    ("pre_normalize", "eval_normalize", "x_vect"),
    [
        (False, False, array([0.1, 0.2, 0.3])),
        (False, True, array([0.55, 0.6, 0.65])),
        (True, False, array([0.1, 0.2, 0.3])),
        (True, True, array([0.55, 0.6, 0.65])),
    ],
)
def test_evaluate_functions_preprocessed(pre_normalize, eval_normalize, x_vect) -> None:
    """Check the evaluation of preprocessed functions."""
    constrained_problem = Power2()
    constrained_problem.preprocess_functions(is_function_input_normalized=pre_normalize)
    values, _ = constrained_problem.evaluate_functions(
        design_vector=x_vect, design_vector_is_normalized=eval_normalize
    )
    assert set(values.keys()) == {"pow2", "ineq1", "ineq2", "eq"}
    assert values["pow2"] == pytest.approx(0.14)
    assert values["ineq1"] == pytest.approx(array([0.499]))
    assert values["ineq2"] == pytest.approx(array([0.492]))
    assert values["eq"] == pytest.approx(array([0.873]))


@pytest.mark.parametrize("preprocess_functions", [False, True])
@pytest.mark.parametrize("no_db_no_norm", [False, True])
@pytest.mark.parametrize(
    ("constraint_names", "keys"), [((), ("g", "h")), (None, ()), (["h"], ("h",))]
)
def test_evaluate_constraints_subset(
    constrained_problem, preprocess_functions, no_db_no_norm, constraint_names, keys
) -> None:
    """Check the evaluation of a subset of constraints."""
    if preprocess_functions:
        constrained_problem.preprocess_functions()

    output_functions, jacobian_functions = constrained_problem.get_functions(
        evaluate_objective=False,
        no_db_no_norm=no_db_no_norm,
        observable_names=None,
        constraint_names=constraint_names,
    )
    values, _ = constrained_problem.evaluate_functions(
        array([0, 0]),
        output_functions=output_functions or None,
        jacobian_functions=jacobian_functions or None,
    )
    assert tuple(values.keys()) == keys


@pytest.mark.parametrize(
    ("observable_names", "keys"),
    [
        (None, {"f", "g", "h"}),
        ((), {"f", "g", "h", "a", "b"}),
        (["a"], {"f", "g", "h", "a"}),
    ],
)
def test_evaluate_observables_subset(
    constrained_problem, observable_names, keys
) -> None:
    """Check the evaluation of a subset of observables."""
    output_functions, jacobian_functions = constrained_problem.get_functions(
        observable_names=observable_names,
    )
    values, _ = constrained_problem.evaluate_functions(
        array([0, 0]),
        output_functions=output_functions,
        jacobian_functions=jacobian_functions,
    )
    assert values.keys() == keys


@pytest.mark.parametrize(
    ("jacobian_names", "keys"),
    [
        (None, set()),
        ((), {"f", "g", "h"}),
        (["h", "b"], {"h", "b"}),
    ],
)
def test_evaluate_jacobians_subset(constrained_problem, jacobian_names, keys) -> None:
    """Check the evaluation of the Jacobian matrices for a subset of the functions."""
    output_functions, jacobian_functions = constrained_problem.get_functions(
        observable_names=None,
        jacobian_names=jacobian_names,
    )
    _, jacobians = constrained_problem.evaluate_functions(
        design_vector=array([0, 0]),
        output_functions=output_functions or None,
        jacobian_functions=jacobian_functions or None,
    )
    assert jacobians.keys() == keys


@pytest.mark.parametrize(
    ("jacobian_names", "message"),
    [(["unknown"], "This name is"), (["other unknown", "unknown"], "These names are")],
)
def test_evaluate_unknown_jacobians(
    constrained_problem, jacobian_names, message
) -> None:
    """Check the evaluation of the Jacobian matrices of unknown functions."""
    with pytest.raises(
        ValueError,
        match=f"{message} not among the names of the functions: "
        f"{', '.join(jacobian_names)}.",
    ):
        constrained_problem.get_functions(jacobian_names=jacobian_names)


@pytest.mark.parametrize(
    ("jacobian_names", "keys"), [(("h"), {"h"}), ([], set()), (None, set())]
)
def test_evaluate_jacobians_alone(constrained_problem, jacobian_names, keys) -> None:
    """Check the evaluation of Jacobian matrices alone."""
    output_functions, jacobian_functions = constrained_problem.get_functions(
        evaluate_objective=False,
        observable_names=None,
        constraint_names=None,
        jacobian_names=jacobian_names,
    )
    values, jacobians = constrained_problem.evaluate_functions(
        design_vector=array([0, 0]),
        output_functions=output_functions or None,
        jacobian_functions=jacobian_functions or None,
    )
    assert not values
    assert jacobians.keys() == keys


def test_no_normalization() -> None:
    problem = Power2()
    OptimizationLibraryFactory().execute(
        problem, algo_name="SLSQP", normalize_design_space=False
    )
    f_opt, _, is_feas, _, _ = problem.optimum
    assert is_feas
    assert abs(f_opt - 2.192) < 0.01


def test_nan_func() -> None:
    problem = Power2()

    def nan_func(_):
        return float("nan")

    problem.objective.func = nan_func
    problem.preprocess_functions()
    with pytest.raises(FunctionIsNan):
        problem.objective.evaluate(zeros(3))


def test_fail_import() -> None:
    with pytest.raises(KeyError):
        OptimizationProblem.from_hdf(FAIL_HDF)


def test_append_export(tmp_wd) -> None:
    """Test the export of an HDF5 file with append mode.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    problem = Rosenbrock()
    problem.preprocess_functions()
    func = problem.objective
    file_path_db = "test_pb_append.hdf5"
    # Export empty file
    problem.to_hdf(file_path_db)

    n_calls = 200
    for i in range(n_calls):
        func.evaluate(array([0.1, 1.0 / (i + 1.0)]))

    # Export again with append mode
    problem.to_hdf(file_path_db, append=True)

    read_db = Database.from_hdf(file_path_db)
    assert len(read_db) == n_calls

    i += 1
    func.evaluate(array([0.1, 1.0 / (i + 1.0)]))

    # Export again with identical elements plus a new one.
    problem.to_hdf(file_path_db, append=True)
    read_db = Database.from_hdf(file_path_db)
    assert len(read_db) == n_calls + 1


def test_grad_normalization(pow2_problem) -> None:
    problem = pow2_problem
    x_vec = ones(3)
    grad = problem.objective.jac(x_vec)
    problem.preprocess_functions()
    norm_grad = problem.objective.jac(x_vec)

    assert pytest.approx(norm(norm_grad - 2 * grad)) == 0.0

    unnorm_grad = problem.design_space.normalize_vect(norm_grad, minus_lb=False)
    assert pytest.approx(norm(unnorm_grad - grad)) == 0.0


def test_2d_objective() -> None:
    disc = SobieskiStructure()
    design_space = SobieskiDesignSpace()
    inputs = disc.io.input_grammar
    design_space.filter([name for name in inputs if not name.startswith("c_")])
    doe_scenario = DOEScenario(
        [disc], "y_12", design_space, formulation_name="DisciplinaryOpt"
    )
    doe_scenario.execute(algo_name="DiagonalDOE", n_samples=10)


def test_observable(pow2_problem) -> None:
    """Test the handling of observables.

    Args:
        pow2_problem: The Power2 problem.
    """
    problem = pow2_problem
    design_norm = "design norm"
    observable = MDOFunction(norm, design_norm)
    problem.add_observable(observable)

    # Check that the observable can be found
    assert problem.observables.get_from_name(design_norm) is observable
    with pytest.raises(ValueError):
        problem.observables.get_from_name("toto")

    # Check that the observable is stored in the database
    OptimizationLibraryFactory().execute(problem, algo_name="SLSQP")
    database = problem.database
    iter_norms = [norm(key.unwrap()) for key in database]
    iter_obs = [value[design_norm] for value in database.values()]
    assert iter_obs == iter_norms

    # Check that the observable is exported
    dataset = problem.to_dataset("dataset")
    func_data = dataset.get_view(group_names="functions").to_dict()
    design_norm_levels = ("functions", design_norm, 0)
    obs_data = func_data.get(design_norm_levels)
    assert obs_data is not None
    assert (
        iter_norms
        == dataset.get_view(group_names="functions", variable_names=design_norm)
        .to_numpy()
        .T
    ).all()
    assert dataset.GRADIENT_GROUP not in dataset.group_names
    dataset = problem.to_dataset("dataset", export_gradients=True)
    assert dataset.GRADIENT_GROUP in dataset.group_names
    name = Database.get_gradient_name("pow2")
    n_iter = len(database)
    n_var = problem.design_space.dimension
    assert dataset.get_view(variable_names=name).shape == (n_iter, n_var)


@pytest.mark.parametrize(
    ("filter_non_feasible", "as_dict", "expected"),
    [
        (
            True,
            True,
            {
                "x": array([
                    [1.0, 1.0, np.power(0.9, 1 / 3)],
                    [0.9, 0.9, np.power(0.9, 1 / 3)],
                ])
            },
        ),
        (
            True,
            False,
            np.array([
                [1.0, 1.0, np.power(0.9, 1 / 3)],
                [0.9, 0.9, np.power(0.9, 1 / 3)],
            ]),
        ),
        (
            False,
            True,
            {
                "x": array([
                    [1.0, 1.0, np.power(0.9, 1 / 3)],
                    [0.9, 0.9, np.power(0.9, 1 / 3)],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                ])
            },
        ),
        (
            False,
            False,
            np.array([
                [1.0, 1.0, np.power(0.9, 1 / 3)],
                [0.9, 0.9, np.power(0.9, 1 / 3)],
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ]),
        ),
    ],
)
def test_get_data_by_names(filter_non_feasible, as_dict, expected) -> None:
    """Test if the data is filtered correctly.

    Args:
        filter_non_feasible: If True, remove the non-feasible points from
                the data.
        as_dict: If True, the data is returned as a dictionary.
        expected: The reference data for the test.
    """
    # Create a Power2 instance
    problem = Power2()
    # Add two feasible points
    problem.database.store(
        np.array([1.0, 1.0, np.power(0.9, 1 / 3)]),
        {"pow2": 2.9, "ineq1": -0.5, "ineq2": -0.5, "eq": 0.0},
    )
    problem.database.store(
        np.array([0.9, 0.9, np.power(0.9, 1 / 3)]),
        {"pow2": 2.55, "ineq1": -0.229, "ineq2": -0.229, "eq": 0.0},
    )
    # Add two non-feasible points
    problem.database.store(
        np.array([0.0, 0.0, 0.0]), {"pow2": 0.0, "ineq1": 0.5, "ineq2": 0.5, "eq": 0.9}
    )
    problem.database.store(
        np.array([0.5, 0.5, 0.5]),
        {"pow2": 0.75, "ineq1": 0.375, "ineq2": 0.375, "eq": 0.775},
    )
    # Get the data back
    data = problem.history.get_data_by_names(
        names=["x"], as_dict=as_dict, filter_non_feasible=filter_non_feasible
    )
    # Check output is filtered when needed
    if as_dict:
        assert np.array_equal(array(list(data.values())).T, expected["x"])
    else:
        assert np.array_equal(data, expected)


def test_gradient_with_random_variables() -> None:
    """Check that the Jacobian is correctly computed with random variable."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("x", "OTUniformDistribution")

    problem = OptimizationProblem(parameter_space)
    problem.objective = MDOFunction(lambda x: 3 * x**2, "func", jac=lambda x: 6 * x)
    PyDOELibrary("PYDOE_FULLFACT").execute(problem, n_samples=3, eval_jac=True)

    data = problem.database.get_gradient_history("func")

    assert array_equal(data, array([0.0, 3.0, 6.0]))


def test_is_mono_objective() -> None:
    """Check the boolean OptimizationProblem.is_mono_objective."""
    design_space = DesignSpace()
    design_space.add_variable("")
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(
        lambda x: array([1.0, 2.0]),
        name="func",
        f_type="obj",
        output_names=["y1", "y2"],
    )

    assert not problem.is_mono_objective

    problem.objective = MDOFunction(
        lambda x: x, name="func", f_type="obj", output_names=["y1"]
    )

    assert problem.is_mono_objective


@pytest.fixture
def problem() -> OptimizationProblem:
    """A simple optimization problem :math:`max_x x`."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0, upper_bound=1, value=0.5)
    opt_problem = OptimizationProblem(design_space)
    opt_problem.objective = MDOFunction(lambda x: x, name="func", f_type="obj")
    return opt_problem


def test_parallel_differentiation(problem) -> None:
    """Check that parallel_differentiation is taken into account."""
    assert not problem.parallel_differentiation
    problem.parallel_differentiation = True
    assert problem.parallel_differentiation


def test_parallel_differentiation_options(problem) -> None:
    """Check that parallel_differentiation_options is taken into account."""
    assert not problem.parallel_differentiation_options
    problem.parallel_differentiation_options = {"step": 1e-10}
    assert problem.parallel_differentiation_options == {"step": 1e-10}


def test_parallel_differentiation_setting_after_functions_preprocessing(
    problem,
) -> None:
    """Check that parallel differentiation cannot be changed after preprocessing."""
    problem.preprocess_functions()
    expected_message = (
        "The parallel differentiation cannot be changed "
        "because the functions have already been pre-processed."
    )
    with pytest.raises(
        RuntimeError,
        match=expected_message,
    ):
        problem.parallel_differentiation = "user"
    with pytest.raises(
        RuntimeError,
        match=expected_message,
    ):
        problem.parallel_differentiation_options = {}


def test_database_name(problem) -> None:
    """Check the name of the database."""
    DOELibraryFactory().execute(problem, algo_name="PYDOE_FULLFACT", n_samples=1)
    problem.database.name = "my_database"
    dataset = problem.to_dataset()
    assert dataset.name == problem.database.name
    dataset = problem.to_dataset("dataset")
    assert dataset.name == "dataset"


@pytest.mark.parametrize(
    ("skip_int_check", "expected_message"),
    [
        (
            True,
            "Forcing the execution of an algorithm that does not handle "
            "integer variables.",
        ),
        (
            False,
            "Algorithm SLSQP is not adapted to the problem, it does not handle "
            "integer variables.\n"
            "Execution may be forced setting the 'skip_int_check' argument "
            "to 'True'.",
        ),
    ],
)
def test_int_opt_problem(skip_int_check, expected_message, caplog) -> None:
    """Test the execution of an optimization problem with integer variables.

    Args:
        skip_int_check: Whether to skip the integer variable handling check
            of the selected algorithm.
        expected_message: The expected message to be recovered from the logger or
            the ValueError message.
        caplog: Fixture to access and control log capturing.
    """
    f_1 = MDOFunction(sin, name="f_1", jac=cos, expr="sin(x)")
    design_space = DesignSpace()
    design_space.add_variable(
        "x", lower_bound=1, upper_bound=3, value=array([1]), type_="integer"
    )
    problem = OptimizationProblem(design_space)
    problem.objective = -f_1

    if skip_int_check:
        OptimizationLibraryFactory().execute(
            problem,
            algo_name="SLSQP",
            normalize_design_space=True,
            skip_int_check=skip_int_check,
        )
        assert expected_message in caplog.text
        assert problem.optimum[1] == array([2.0])
    else:
        with pytest.raises(ValueError, match=expected_message):
            OptimizationLibraryFactory().execute(
                problem,
                algo_name="SLSQP",
                normalize_design_space=True,
                skip_int_check=skip_int_check,
            )


@pytest.fixture
def constrained_problem() -> OptimizationProblem:
    """A constrained optimisation problem with multidimensional constraints."""
    design_space = DesignSpace()
    design_space.add_variable("x", 2, value=1.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x.sum(), "f", jac=lambda x: [1, 1])
    problem.add_constraint(
        MDOFunction(operator.itemgetter(0), "g", jac=lambda _: [1, 0], dim=1),
        constraint_type="ineq",
    )
    problem.add_constraint(
        MDOFunction(lambda x: x, "h", jac=lambda _: np.eye(2)), constraint_type="eq"
    )
    problem.add_observable(
        MDOFunction(operator.itemgetter(1), "a", jac=lambda x: [0, 1])
    )
    problem.add_observable(MDOFunction(operator.neg, "b", jac=lambda x: -np.eye(2)))
    return problem


@pytest.mark.parametrize(
    ("names", "dimensions"),
    [(None, {"f": 1, "g": 1, "h": 2}), (["g", "h"], {"g": 1, "h": 2})],
)
def test_get_functions_dimensions(constrained_problem, names, dimensions) -> None:
    """Check the computation of the functions dimensions."""
    assert constrained_problem.get_functions_dimensions(names) == dimensions


parametrize_unsatisfied_constraints = pytest.mark.parametrize(
    ("design", "n_unsatisfied"),
    [
        (array([0.0, 0.0]), 0),
        (array([-1.0, 0.0]), 1),
        (array([-1.0, -1.0]), 2),
        (array([1.0, 1.0]), 3),
    ],
)


@parametrize_unsatisfied_constraints
def test_get_number_of_unsatisfied_constraints_from_passed_values(
    constrained_problem, design, n_unsatisfied
) -> None:
    """Check the computation of the number of unsatisfied constraints from values."""
    constrained_problem.evaluate_functions = mock.Mock()
    assert (
        constrained_problem.constraints.get_number_of_unsatisfied_constraints({
            "g": design[0],
            "h": design,
        })
        == n_unsatisfied
    )
    constrained_problem.evaluate_functions.assert_not_called()


def test_scalar_constraint_names(constrained_problem) -> None:
    """Check the computation of the scalar constraints names."""
    scalar_names = constrained_problem.scalar_constraint_names
    assert set(scalar_names) == {
        "g",
        "h[0]",
        "h[1]",
    }


def test_observables_callback() -> None:
    """Test that the observables are called properly."""
    problem = Power2()
    obs1 = MDOFunction(norm, "design_norm")
    problem.add_observable(obs1)
    problem.database.store(
        array([0.79499653, 0.20792012, 0.96630481]),
        {"pow2": 1.61, "ineq1": -0.0024533, "ineq2": -0.0024533, "eq": -0.00228228},
    )

    problem.preprocess_functions(is_function_input_normalized=False)
    problem.observables.evaluate(array([0.79499653, 0.20792012, 0.96630481]))

    assert problem.observables[0].n_calls == 1


def test_approximated_jacobian_wrt_uncertain_variables() -> None:
    """Check that the approximated Jacobian wrt uncertain variables is correct."""
    uspace = ParameterSpace()
    uspace.add_random_variable("u", "OTNormalDistribution")
    problem = OptimizationProblem(uspace)
    problem.differentiation_method = problem.ApproximationMode.FINITE_DIFFERENCES
    problem.objective = MDOFunction(lambda u: u, "func")
    CustomDOE().execute(problem, samples=array([[0.0]]), eval_jac=True)
    grad = problem.database.get_gradient_history("func")
    assert grad[0, 0] == pytest.approx(1.0, abs=1e-3)


@pytest.fixture
def rosenbrock_lhs() -> tuple[Rosenbrock, dict[str, ndarray]]:
    """The Rosenbrock problem after evaluation and its start point."""
    problem = Rosenbrock()
    problem.add_observable(MDOFunction(sum, "obs"))
    problem.add_constraint(MDOFunction(sum, "cstr"), constraint_type="ineq")
    start_point = problem.design_space.get_current_value(as_dict=True)
    execute_algo(problem, algo_name="LHS", n_samples=3, algo_type="doe")
    return problem, start_point


def test_reset(rosenbrock_lhs) -> None:
    """Check the default behavior of OptimizationProblem.reset."""
    problem, start_point = rosenbrock_lhs
    nonproc_functions = [
        problem.objective.original,
        *problem.constraints.get_originals(),
        *problem.observables.get_originals(),
        *problem.new_iter_observables.get_originals(),
    ]
    problem.reset()
    assert len(problem.database) == 0
    assert id(problem.objective.original) == id(problem.objective)
    for key, val in problem.design_space.get_current_value(as_dict=True).items():
        assert (start_point[key] == val).all()

    functions = [
        problem.objective,
        *problem.constraints,
        *problem.observables,
        *problem.new_iter_observables,
    ]
    for func, nonproc_func in zip(functions, nonproc_functions):
        assert id(func) == id(nonproc_func)
        assert func.n_calls == 0
        assert nonproc_func.n_calls == 0

    nonproc_functions = [
        problem.objective.original,
        *problem.constraints.get_originals(),
        *problem.observables.get_originals(),
        *problem.new_iter_observables.get_originals(),
    ]
    functions = [
        problem.objective,
        *problem.constraints,
        *problem.observables,
        *problem.new_iter_observables,
    ]
    for func, nonproc_func in zip(functions, nonproc_functions):
        assert id(func) == id(nonproc_func)


def test_reset_database(rosenbrock_lhs) -> None:
    """Check OptimizationProblem.reset without database reset."""
    problem, _ = rosenbrock_lhs
    problem.reset(database=False)
    assert len(problem.database) == 3


def test_reset_current_iter(rosenbrock_lhs) -> None:
    """Check OptimizationProblem.reset without current_iter reset."""
    problem, _ = rosenbrock_lhs
    problem.reset(current_iter=False)
    assert len(problem.database) == 0


def test_reset_design_space(rosenbrock_lhs) -> None:
    """Check OptimizationProblem.reset without design_space reset."""
    problem, start_point = rosenbrock_lhs
    problem.reset(design_space=False)
    for key, val in problem.design_space.get_current_value(as_dict=True).items():
        assert (start_point[key] != val).any()


def test_reset_functions(rosenbrock_lhs) -> None:
    """Check OptimizationProblem.reset without reset the number of function calls."""
    problem, _ = rosenbrock_lhs
    problem.reset(function_calls=False)
    assert problem.objective.n_calls == 3


def test_reset_wo_current_value() -> None:
    """Check OptimizationProblem.reset when the default design value is missing."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "obj")
    problem.design_space.set_current_value({"x": array([0.0])})
    problem.reset()
    assert problem.design_space.get_current_value(as_dict=True) == {}


def test_reset_preprocess(rosenbrock_lhs) -> None:
    """Check OptimizationProblem.reset without functions pre-processing reset."""
    problem, _ = rosenbrock_lhs
    problem.reset(preprocessing=False)
    assert id(problem.objective) != id(problem.objective.original)
    functions = [
        problem.objective,
        *problem.constraints,
        *problem.observables,
        *problem.new_iter_observables,
    ]
    nonproc_functions = [
        problem.objective.original,
        *problem.constraints.get_originals(),
        *problem.observables.get_originals(),
        *problem.new_iter_observables.get_originals(),
    ]
    for func, nonproc_func in zip(functions, nonproc_functions):
        assert id(func) != id(nonproc_func)

    assert problem.objective.original is not None
    assert len(tuple(problem.constraints.get_originals())) == len(problem.constraints)
    assert len(tuple(problem.observables.get_originals())) == len(problem.observables)
    assert len(tuple(problem.new_iter_observables.get_originals())) == len(
        problem.new_iter_observables
    )


def test_function_string_representation_from_hdf() -> None:
    """Check the string representation of a function when importing a HDF5 file.

    The commented code is the one used for creating the HDF5 file.
    """
    # design_space = DesignSpace()
    # design_space.add_variable("x0", lower_bound=0.0, upper_bound=1.0, value=0.5)
    # design_space.add_variable("x1", lower_bound=0.0, upper_bound=1.0, value=0.5)
    # problem = OptimizationProblem(design_space)
    # problem.objective = MDOFunction(
    #     lambda x: x[0] + x[1], "f", input_names=["x0", "x1"]
    # )
    # problem.constraints.append(
    #     MDOFunction(
    #         lambda x: x[0] + x[1],
    #         "g",
    #         input_names=["x0", "x1"],
    #         f_type=MDOFunction.ConstraintType.INEQ,
    #     )
    # )
    # problem.to_hdf("opt_problem_to_check_string_representation.hdf5")
    new_problem = OptimizationProblem.from_hdf(
        DIRNAME / "opt_problem_to_check_string_representation.hdf5"
    )
    assert str(new_problem.objective) == "f(x0, x1)"
    assert str(new_problem.constraints[0]) == "g(x0, x1) <= 0.0"


@pytest.mark.parametrize(("name", "dimension"), [("f", 1), ("g", 1), ("h", 2)])
def test_get_function_dimension(constrained_problem, name, dimension) -> None:
    """Check the output dimension of a problem function."""
    assert constrained_problem.get_function_dimension(name) == dimension


def test_get_function_dimension_unknown(constrained_problem) -> None:
    """Check the output dimension of an unknown problem function."""
    with pytest.raises(
        ValueError, match=re.escape("The problem has no function named unknown.")
    ):
        constrained_problem.get_function_dimension("unknown")


@pytest.fixture
def design_space() -> mock.Mock:
    """A design space."""
    design_space = mock.Mock()
    design_space.get_current_x = mock.Mock()
    return design_space


@pytest.fixture
def function() -> mock.Mock:
    """A function."""
    function = mock.Mock()
    function.evaluate = mock.MagicMock(return_value=1.0)
    function.name = "f"
    return function


@pytest.mark.parametrize("expects_normalized", [(True,), (False,)])
def test_get_function_dimension_no_dim(
    function, design_space, expects_normalized
) -> None:
    """Check the implicitly defined output dimension of a problem function."""
    function.dim = 0
    function.expects_normalized_inputs = expects_normalized
    design_space.has_current_value = mock.Mock(return_value=True)
    problem = OptimizationProblem(design_space)
    problem.objective = function
    design_space.get_current_value = mock.Mock()
    assert problem.get_function_dimension(function.name) == 1
    if expects_normalized:
        design_space.get_current_value.assert_called_once()
    else:
        design_space.get_current_value.assert_not_called()
        assert design_space.get_current_value.call_count == 2


def test_get_function_dimension_unavailable(function, design_space) -> None:
    """Check the unavailable output dimension of a problem function."""
    function.dim = 0
    design_space.has_current_value = False
    problem = OptimizationProblem(design_space)
    problem.objective = function
    with pytest.raises(
        RuntimeError,
        match=f"The output dimension of function {function.name} is not available.",
    ):
        problem.get_function_dimension(function.name)


@pytest.mark.parametrize("categorize", [True, False])
@pytest.mark.parametrize("export_gradients", [True, False])
def test_dataset_missing_values(categorize, export_gradients) -> None:
    """Test the export of a database with missing values to a dataset.

    Args:
        categorize: If True, remove the non-feasible points from
                the data.
        export_gradients: If True, export the gradient to the dataset.
    """
    problem = Power2()
    # Add a complete evaluation.
    problem.database.store(
        np.array([1.0, 1.0, 1.0]),
        {
            "pow2": 3.0,
            "design norm": 1.7320508075688772,
            "@pow2": array([2.0, 2.0, 2.0]),
        },
    )
    # Add a point with missing values.
    problem.database.store(np.array([-1.0, -1.0, -1.0]), {})
    # Add a complete evaluation.
    problem.database.store(
        np.array([-1.77635684e-15, 1.33226763e-15, 4.44089210e-16]),
        {
            "pow2": 5.127595883936577e-30,
            "design norm": 2.2644195468014703e-15,
            "@pow2": array([-3.55271368e-15, 2.66453526e-15, 8.88178420e-16]),
        },
    )
    # Add one evaluation with complete function data but missing gradient.
    problem.database.store(np.array([0.0, 0.0, 0.0]), {"pow2": 0.0, "design norm": 0.0})
    # Another point with missing values.
    problem.database.store(np.array([0.5, 0.5, 0.5]), {})
    # Export to a dataset.
    dataset = problem.to_dataset(
        categorize=categorize, export_gradients=export_gradients
    )
    # Check that the missing values are exported as NaN.
    if categorize:
        if export_gradients:
            assert (
                dataset.get_view(group_names="functions", indices=4).to_numpy()
                == np.array([0.0, 0.0])
            ).all()
            assert (
                dataset.get_view(group_names="gradients", indices=4)
                .isnull()
                .to_numpy()
                .all()
            )

        else:
            assert (
                dataset.get_view(group_names="functions", indices=2)
                .isnull()
                .to_numpy()
                .all()
            )

    elif export_gradients:
        assert array_equal(
            dataset.get_view(group_names="parameters", indices=3).to_numpy()[0, :],
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan]),
            equal_nan=True,
        )

    else:
        assert array_equal(
            dataset.get_view(group_names="parameters", indices=4).to_numpy()[0, :],
            np.array([0.5, 0.5, 0.5, np.nan, np.nan]),
            equal_nan=True,
        )


@pytest.fixture
def problem_for_eval_obs_jac() -> OptimizationProblem:
    """An optimization problem to check the option eval_obs_jac."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.0)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: 1 - x, "f", jac=lambda x: array([[-1.0]]))
    problem.add_constraint(
        MDOFunction(lambda x: x - 0.5, "c", f_type="ineq", jac=lambda x: array([[1.0]]))
    )
    problem.add_observable(MDOFunction(lambda x: x, "o", jac=lambda x: array([[1.0]])))
    return problem


@pytest.mark.parametrize(
    "options",
    [
        {"algo_name": "SLSQP", "algo_type": "opt", "max_iter": 1},
        {"algo_name": "PYDOE_FULLFACT", "algo_type": "doe", "n_samples": 1},
    ],
)
@pytest.mark.parametrize("eval_obs_jac", [True, False])
@pytest.mark.parametrize("store_jacobian", [True, False])
def test_jabobian_in_database(
    problem_for_eval_obs_jac, options, eval_obs_jac, store_jacobian
) -> None:
    """Check Jacobian matrices in database in function of eval_obs_jac and
    store_jacobian options.
    """
    problem_for_eval_obs_jac.reset()
    execute_algo(
        problem_for_eval_obs_jac,
        eval_obs_jac=eval_obs_jac,
        store_jacobian=store_jacobian,
        **options,
    )
    database = problem_for_eval_obs_jac.database
    function_names = database.get_function_names(False)
    assert ("@o" in function_names) is (eval_obs_jac and store_jacobian)
    store_f_and_c = store_jacobian and options["algo_name"] == "SLSQP"
    assert ("@f" in function_names) is store_f_and_c
    assert ("@c" in function_names) is store_f_and_c


def test_presence_observables_hdf_file(pow2_problem, tmp_wd) -> None:
    """Check if the observables can be retrieved in an HDF file after export and
    import.
    """
    # Add observables to the optimization problem.
    obs1 = MDOFunction(norm, "design norm")
    pow2_problem.add_observable(obs1)
    obs2 = MDOFunction(sum, "sum")
    pow2_problem.add_observable(obs2)

    OptimizationLibraryFactory().execute(pow2_problem, algo_name="SLSQP")

    # Export and import the optimization problem.
    file_path = "power2.h5"
    pow2_problem.to_hdf(file_path)
    imp_pb = OptimizationProblem.from_hdf(file_path)

    # Check the set of observables.
    # Assuming that two functions are equal if they have the same name.
    exp_obs_names = {obs.name for obs in pow2_problem.observables}
    imp_obs_names = {obs.name for obs in imp_pb.observables}
    assert exp_obs_names == imp_obs_names


@pytest.mark.parametrize(
    ("input_values", "expected"),
    [
        ((), array([[1.0], [2.0]])),
        (array([[1.0], [2.0], [1.0]]), array([[1.0], [2.0], [1.0]])),
    ],
)
def test_export_to_dataset(input_values, expected) -> None:
    """Check the export of the database."""
    design_space = DesignSpace()
    design_space.add_variable("dv")

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x * 2, "obj")
    problem.add_constraint(
        MDOFunction(lambda x: x * 3, "cstr", f_type=MDOFunction.ConstraintType.INEQ)
    )

    algo = CustomDOE()
    algo.execute(problem, samples=array([[1.0], [2.0], [1.0]]))

    dataset = problem.to_dataset(input_values=input_values)

    assert dataset.misc["input_space"] == design_space
    assert id(dataset.misc["input_space"]) != id(design_space)

    assert_equal(dataset.get_view(variable_names="dv").to_numpy(), expected)
    assert_equal(dataset.get_view(variable_names="obj").to_numpy(), expected * 2)
    assert_equal(dataset.get_view(variable_names="cstr").to_numpy(), expected * 3)


@pytest.mark.skip("Input names are sorted.")
@pytest.mark.parametrize("name", ["a", "c"])
def test_export_to_dataset_input_names_order(name) -> None:
    """Check that the order of the input names is not changed in the dataset."""
    design_space = DesignSpace()
    design_space.add_variable("b")
    design_space.add_variable(name)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x[0] + x[1], "obj")

    algo = CustomDOE()
    algo.execute(problem, samples=array([[1.0, 1.0], [2.0, 2.0]]))

    dataset = problem.to_dataset()
    assert dataset.get_variable_names("design_parameters") == ["b", name]


@pytest.fixture(scope="module")
def problem_with_complex_value() -> OptimizationProblem:
    """A problem using a design space with a float variable whose value is complex."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.set_current_value({"x": array([1.0 + 0j])})
    return OptimizationProblem(design_space)


@pytest.mark.parametrize(
    ("cast_to_real", "as_dict", "x0"),
    [
        (False, False, [1.0 + 0j]),
        (False, True, {"x": 1.0 + 0j}),
        (True, False, [1.0]),
        (True, True, {"x": [1.0]}),
    ],
)
def test_get_x0_normalized_complex(
    problem_with_complex_value, cast_to_real, as_dict, x0
) -> None:
    """Check the getting of a normalized complex initial value."""
    assert_equal(
        problem_with_complex_value.design_space.get_current_value(
            complex_to_real=cast_to_real, as_dict=as_dict, normalize=True
        ),
        x0,
    )


def test_objective_name() -> None:
    """Check the name of the objective."""
    problem = OptimizationProblem(DesignSpace())
    problem.objective = MDOFunction(lambda x: x, "f")
    assert problem.standardized_objective_name == "f"
    assert problem.objective_name == "f"
    problem.minimize_objective = False
    assert problem.standardized_objective_name == "-f"
    assert problem.objective_name == "f"


@pytest.mark.parametrize(
    "cstr_type", [MDOFunction.ConstraintType.EQ, MDOFunction.FunctionType.INEQ]
)
@pytest.mark.parametrize("has_default_name", [False, True])
@pytest.mark.parametrize(
    ("value", "positive", "name"),
    [
        (0.0, False, "c"),
        (0.0, True, "-c"),
        (1.0, True, "-[c-1.0]"),
        (-1.0, True, "-[c+1.0]"),
        (1.0, False, "[c-1.0]"),
        (-1.0, False, "[c+1.0]"),
    ],
)
def test_original_to_current_names(
    has_default_name, value, positive, cstr_type, name
) -> None:
    """Check the name of a constraint."""
    problem = OptimizationProblem(DesignSpace())
    original_name = "c"
    constraint_function = MDOFunction(lambda x: x, original_name)
    constraint_function.has_default_name = has_default_name
    problem.add_constraint(
        constraint_function,
        value=value,
        positive=positive,
        constraint_type=cstr_type,
    )
    cstr_name = problem.constraints[0].name

    assert problem.constraints.original_to_current_names[original_name] == [cstr_name]

    if has_default_name:
        assert cstr_name == name
    else:
        assert cstr_name == "c"


def test_original_to_current_names_with_aggregation() -> None:
    """Check the name of the constraints when some are aggregated."""
    problem = OptimizationProblem(DesignSpace())
    nb_constr = 5
    for i in range(nb_constr):
        original_name = f"c{i}"
        problem.add_constraint(
            MDOFunction(lambda x: x, original_name),
            constraint_type=MDOFunction.FunctionType.INEQ,
        )

    idx_aggr = [0, 3]
    for i in idx_aggr:
        problem.constraints.aggregate(
            i, method=OptimizationProblem.AggregationFunction.MAX
        )

    for i in range(nb_constr):
        name = f"c{i}"
        if i in idx_aggr:
            assert problem.constraints.original_to_current_names[name] == [f"max_c{i}"]
        else:
            assert problem.constraints.original_to_current_names[name] == [name]


def test_observables_normalization(sellar_disciplines) -> None:
    """Test that the observables are called at each iteration."""
    design_space = DesignSpace()
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=10.0, value=ones(1))
    design_space.add_variable(
        "x_shared",
        2,
        lower_bound=(-10, 0.0),
        upper_bound=(10.0, 10.0),
        value=array([4.0, 3.0]),
    )
    scenario = create_scenario(
        sellar_disciplines,
        "obj",
        design_space,
        formulation_name="MDF",
    )
    scenario.add_constraint("c_1", constraint_type="ineq")
    scenario.add_constraint("c_2", constraint_type="ineq")
    scenario.add_observable("y_1")
    scenario.execute(algo_name="SLSQP", max_iter=3)
    total_iter = len(scenario.formulation.optimization_problem.database)
    n_obj_eval = (
        scenario.formulation.optimization_problem.database.get_function_history(
            "y_1"
        ).size
    )
    n_obs_eval = (
        scenario.formulation.optimization_problem.database.get_function_history(
            "obj"
        ).size
    )
    assert total_iter == n_obj_eval == n_obs_eval


def test_observable_cannot_be_added_twice(caplog) -> None:
    """Check that an observable cannot be added twice."""
    problem = OptimizationProblem(DesignSpace())
    problem.add_observable(MDOFunction(lambda x: x, "obs"))
    problem.add_observable(MDOFunction(lambda x: x, "obs"))
    assert "WARNING" in caplog.text
    assert 'The optimization problem already observes "obs".' in caplog.text
    assert len(problem.observables) == 1


def test_repr_constraint_linear_lower_ineq() -> None:
    """Check the representation of a linear lower inequality-constraint."""
    design_space = DesignSpace()
    design_space.add_variable("x", 2)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOLinearFunction(array([1, 2]), "f")
    problem.add_constraint(
        MDOLinearFunction(
            array([[0, 1], [2, 3], [4, 5]]),
            "g",
            value_at_zero=array([6, 7, 8]),
            f_type=MDOLinearFunction.ConstraintType.INEQ,
        ),
        positive=True,
    )
    assert str(problem) == (
        """Optimization problem:
   minimize f(x[0], x[1]) = x[0] + 2.00e+00*x[1]
   with respect to x
   subject to constraints:
      g(x[0], x[1]): [ 0.00e+00  1.00e+00][x[0]] + [ 6.00e+00] >= 0.0
                     [ 2.00e+00  3.00e+00][x[1]]   [ 7.00e+00]
                     [ 4.00e+00  5.00e+00]         [ 8.00e+00]"""
    )


def test_get_original_observable(pow2_problem) -> None:
    """Check the accessor to an original observable."""
    function = MDOFunction(None, "f")
    pow2_problem.add_observable(function)
    assert pow2_problem.observables.get_from_name(function.name) is function


def test_get_preprocessed_observable(pow2_problem) -> None:
    """Check the accessor to a pre-processed observable."""
    function = MDOFunction(None, "f")
    pow2_problem.add_observable(function)
    pow2_problem.preprocess_functions()
    assert (
        pow2_problem.observables.get_from_name(function.name)
        is pow2_problem.observables[-1]
    )


def test_get_missing_observable(constrained_problem) -> None:
    """Check the accessor to a missing observable."""
    match = "missing_observable_name is not among the names of the observables: a, b."
    with pytest.raises(ValueError, match=match):
        constrained_problem.observables.get_from_name("missing_observable_name")


@pytest.mark.parametrize("name", ["obj", "cstr"])
def test_execute_twice(problem_executed_twice, name) -> None:
    """Check that the second evaluations of an OptimizationProblem works."""
    assert len(problem_executed_twice.database.get_function_history(name)) == 2


def test_avoid_complex_in_dataset() -> None:
    """Check that exporting database to dataset casts complex numbers to real."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(
        lambda x: array([0j]), "f", jac=lambda x: array([[0j]])
    )
    problem.preprocess_functions()
    output_functions, jacobian_functions = problem.get_functions(jacobian_names=())
    problem.evaluate_functions(
        array([0.25 + 0j]),
        output_functions=output_functions,
        jacobian_functions=jacobian_functions,
    )
    dataset = problem.to_dataset(export_gradients=True)
    for name in ["@f", "f", "x"]:
        assert dataset.get_view(variable_names=name).to_numpy().dtype.kind == "f"


@pytest.mark.parametrize(
    "cstr_type", [MDOFunction.ConstraintType.EQ, MDOFunction.ConstraintType.INEQ]
)
@pytest.mark.parametrize(
    ("is_feasible", "violation", "cstr"),
    [
        (False, float("inf"), [float("NaN")]),
        (True, 0.0, [0.0, 0.0]),
        (True, 0.0, [1.0, 1.0]),
        (False, 0.01, [1.0, 1.1]),
        (False, 4.0, [3.0, 0.0]),
        (False, 13.0, [3.0, 4.0]),
    ],
)
def test_check_design_point_is_feasible(
    cstr_type, is_feasible, violation, cstr
) -> None:
    """Test check_design_point_is_feasible."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.tolerances.inequality = 1
    problem.tolerances.equality = 1
    problem.objective = MDOFunction(lambda x: x, "obj")
    problem.add_constraint(MDOFunction(lambda x: x, "cstr"), constraint_type=cstr_type)

    x_vect = array([1.0])
    problem.database.store(x_vect, {"obj": array([1]), "cstr": array(cstr)})
    assert problem.history.check_design_point_is_feasible(x_vect) == pytest.approx((
        is_feasible,
        violation,
    ))


def test_is_multi_objective() -> None:
    assert not BinhKorn().is_mono_objective

    design_space = create_design_space()
    design_space.add_variable("x", 1)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: array([x, x]), "two")
    with pytest.raises(
        ValueError, match=re.escape("Cannot determine the dimension of the objective.")
    ):
        problem.is_mono_objective  # noqa: B018

    problem.objective.evaluate(1.0)
    assert not problem.is_mono_objective

    problem.objective.dim = 0
    with pytest.raises(
        ValueError, match=re.escape("Cannot determine the dimension of the objective.")
    ):
        problem.is_mono_objective  # noqa: B018

    problem.objective.output_names = ["x", "x"]
    assert not problem.is_mono_objective


def test_optimization_result_save_nested_dict(tmp_wd) -> None:
    """Check that the nested dictionaries of OptimizationResult are correctly saved."""
    problem = Power2()
    execute_algo(problem, algo_name="SLSQP")
    problem.to_hdf("problem.hdf5")
    x_0_as_dict = problem.solution.x_0_as_dict
    x_opt_as_dict = problem.solution.x_opt_as_dict
    problem = OptimizationProblem.from_hdf("problem.hdf5")
    assert compare_dict_of_arrays(x_0_as_dict, problem.solution.x_0_as_dict)
    assert compare_dict_of_arrays(x_opt_as_dict, problem.solution.x_opt_as_dict)


@pytest.mark.parametrize(
    ("hdf_node_path", "expected"), [("", ""), ("node_name", " at node node_name")]
)
def test_to_from_hdf_log(pow2_problem, caplog, tmp_wd, hdf_node_path, expected):
    """Check log when using to_hdf and from_hdf."""
    file_path = "problem.hdf5"
    pow2_problem.to_hdf(file_path, hdf_node_path=hdf_node_path)
    pow2_problem.from_hdf(file_path, hdf_node_path=hdf_node_path)
    assert caplog.record_tuples[0] == (
        "gemseo.algos.optimization_problem",
        20,
        "Exporting the optimization problem to the file problem.hdf5" + expected,
    )
    assert caplog.record_tuples[1] == (
        "gemseo.algos.optimization_problem",
        20,
        "Importing the optimization problem from the file problem.hdf5" + expected,
    )


def test_hdf_node_path(pow2_problem, tmp_wd):
    """Check the importation/exportation in a specific node."""
    file_name = "test_hdf_node.hdf5"
    node = "problem_node"
    problem = pow2_problem
    function_names = problem.function_names
    desvar_names = problem.design_space.variable_names
    problem.to_hdf(file_name, hdf_node_path=node)

    # Should fail : no opt_problem saved at the root
    with pytest.raises(KeyError):
        OptimizationProblem.from_hdf(file_name)

    # Should fail: node doesn't exist
    with pytest.raises(KeyError):
        OptimizationProblem.from_hdf(file_name, hdf_node_path="wrong_node")

    # Should succeed : check if functions and design variables are correct
    imp_prob = OptimizationProblem.from_hdf(file_name, hdf_node_path=node)
    assert_equal(imp_prob.function_names, function_names)
    assert_equal(imp_prob.design_space.variable_names, desvar_names)

    # Test saving options in nested dict form
    algo_opts = {"sub_algorithm_settings": {"eq_tolerance": 1e-1}}
    execute_algo(
        problem,
        algo_name="Augmented_Lagrangian_order_0",
        sub_algorithm_name="SLSQP",
        **algo_opts,
    )
    problem.to_hdf(file_name, hdf_node_path=node)

    # Test import
    # Should succeed : check if functions and design variables are correct
    imp_prob = OptimizationProblem.from_hdf(file_name, hdf_node_path=node)
    assert_equal(imp_prob.function_names, function_names)
    assert_equal(imp_prob.design_space.variable_names, desvar_names)


def test_repr_html():
    """Check the string and HTML representation of an optimization problem."""
    problem = Power2()
    assert (
        repr(problem)
        == str(problem)
        == """Optimization problem:
   minimize pow2(x) = x[0]**2 + x[1]**2 + x[2]**2
   with respect to x
   subject to constraints:
      ineq1(x): 0.5 - x[0]**3 <= 0.0
      ineq2(x): 0.5 - x[1]**3 <= 0.0
      eq(x): 0.9 - x[2]**3 == 0.0"""
    )
    assert problem._repr_html_() == REPR_HTML_WRAPPER.format(
        "Optimization problem:<br/>"
        "<ul>"
        "<li>minimize pow2(x) = x[0]**2 + x[1]**2 + x[2]**2</li>"
        "<li>with respect to x</li><li>subject to constraints:"
        "<ul>"
        "<li>ineq1(x): 0.5 - x[0]**3 &lt;= 0.0</li>"
        "<li>ineq2(x): 0.5 - x[1]**3 &lt;= 0.0</li>"
        "<li>eq(x): 0.9 - x[2]**3 == 0.0</li>"
        "</ul>"
        "</li>"
        "</ul>"
    )


@pytest.mark.parametrize("minimize", [True, False])
def test_minimize_objective(pow2_problem, minimize) -> None:
    """Test the minimize objective setter."""
    initial_minimize = pow2_problem.minimize_objective
    x_0 = np.ones(3)
    f_0 = pow2_problem.objective.evaluate(x_0)

    pow2_problem.minimize_objective = minimize
    f_1 = pow2_problem.objective.evaluate(x_0)

    assert pow2_problem.minimize_objective == minimize

    if initial_minimize == minimize:
        assert f_0 == f_1
    else:
        assert f_0 == -f_1


def test_reformulate_with_slack_variables(constrained_problem) -> None:
    """Test the reformulation of the optimization problem with slack variables."""
    reformulated_problem = (
        constrained_problem.get_reformulated_problem_with_slack_variables()
    )
    assert (
        len(tuple(reformulated_problem.constraints.get_inequality_constraints())) == 0
    )
    assert (
        reformulated_problem.design_space.dimension
        == constrained_problem.design_space.dimension
        + next(constrained_problem.constraints.get_inequality_constraints()).dim
    )
    assert len(
        list(reformulated_problem.constraints.get_equality_constraints())
    ) == len(tuple(reformulated_problem.constraints.get_equality_constraints())) + len(
        list(reformulated_problem.constraints.get_inequality_constraints())
    )


@pytest.mark.parametrize("value", [0.5, None])
@pytest.mark.parametrize(
    "differentiation_method",
    [
        OptimizationProblem.DifferentiationMethod.FINITE_DIFFERENCES,
        OptimizationProblem.DifferentiationMethod.COMPLEX_STEP,
    ],
)
def test_no_initial_value_with_approximated_gradient(value, differentiation_method):
    """Check that gradient approximation works with and without current value."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=value)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x**2, "f")
    problem.differentiation_method = differentiation_method
    optimization_result = execute_algo(problem, algo_name="SLSQP", max_iter=100)
    assert optimization_result.x_opt == 0


@pytest.mark.parametrize("preprocess", [False, True])
def test_get_all_functions(preprocess):
    """Check get_all_functions."""
    problem = Rosenbrock()
    if preprocess:
        problem.preprocess_functions()

    functions = problem.functions
    assert functions == [
        problem.objective,
        *problem.constraints,
        *problem.observables,
    ]

    functions = problem.original_functions
    if preprocess:
        assert functions == [
            problem.objective.original,
            *problem.constraints.get_originals(),
            *problem.observables.get_originals(),
        ]
    else:
        assert functions == [
            problem.objective,
            *problem.constraints,
            *problem.observables,
        ]


def test_observables_setters():
    """Check that the observables setters work properly."""
    problem = EvaluationProblem(DesignSpace())
    f_type = MDOFunction.FunctionType.OBS
    functions = [MDOFunction(lambda x: x, name, f_type=f_type) for name in "fg"]
    problem.observables = functions
    problem.new_iter_observables = functions

    # Keep the previous lines
    # to check that observables and new_iter_observables are cleared.
    functions = [MDOFunction(lambda x: x, name, f_type=f_type) for name in "hi"]
    problem.observables = functions
    problem.new_iter_observables = functions
    assert problem.observables._functions == functions
    assert problem.new_iter_observables._functions == functions


@pytest.mark.parametrize("output_name", ["x", "f"])
def test_evaluation_problem_to_dataset(output_name):
    """Check EvaluationProblem.to_dataset."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=2.0)
    problem = EvaluationProblem(design_space)
    problem.add_observable(MDOFunction(lambda x: 2 * x, output_name))
    problem.preprocess_functions()
    output_functions = problem.get_functions(observable_names=())[0]
    problem.evaluate_functions(
        array([1.0]),
        design_vector_is_normalized=False,
        output_functions=output_functions or None,
    )
    problem.evaluate_functions(
        array([2.0]),
        design_vector_is_normalized=False,
        output_functions=output_functions or None,
    )

    dataset = IODataset()
    dataset.add_input_variable("x", array([[1.0], [2.0]]))
    dataset.add_output_variable(output_name, array([[2.0], [4.0]]))
    assert_frame_equal(problem.to_dataset(), dataset)

    dataset = Dataset()
    if output_name == "f":
        dataset.add_variable("x", array([[1.0], [2.0]]))
        dataset.add_variable(output_name, array([[2.0], [4.0]]))
    else:
        dataset.add_variable("x", array([[1.0, 2.0], [2.0, 4.0]]))
        dataset.columns = MultiIndex.from_tuples(
            [("parameters", "x", 0), ("parameters", "x", 0)],
            names=["GROUP", "VARIABLE", "COMPONENT"],
        )
    assert_frame_equal(problem.to_dataset(categorize=False), dataset)


@pytest.fixture
def evaluation_problem() -> EvaluationProblem:
    """An evaluation problem."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    problem = EvaluationProblem(design_space)
    problem.add_observable(
        MDOFunction(lambda x: 2 * x, "f", jac=lambda x: array([2.0]))
    )
    return problem


@pytest.mark.parametrize("design_vector_is_normalized", [False, True])
@pytest.mark.parametrize(
    ("name1", "name2", "i"),
    [
        ("observable_names", "output_functions", 0),
        ("jacobian_names", "jacobian_functions", 1),
    ],
)
def test_max_iter_reached_exception(
    evaluation_problem, design_vector_is_normalized, name1, name2, i
):
    """Check MaxIterReachedException."""
    evaluation_problem.preprocess_functions()
    functions = evaluation_problem.get_functions(**{name1: ("f",)})[i]
    evaluation_problem.evaluation_counter.maximum = 2
    kwargs = {name2: functions}
    evaluation_problem.evaluate_functions(
        array([0.1]), design_vector_is_normalized=design_vector_is_normalized, **kwargs
    )
    evaluation_problem.evaluate_functions(
        array([0.2]), design_vector_is_normalized=design_vector_is_normalized, **kwargs
    )
    evaluation_problem.evaluation_counter.current = 2
    with pytest.raises(MaxIterReachedException):
        evaluation_problem.evaluate_functions(
            array([0.3]),
            design_vector_is_normalized=design_vector_is_normalized,
            **kwargs,
        )


def test_stop_if_nan(evaluation_problem):
    """Check stop_if_nan when the functions are not ProblemFunction."""
    evaluation_problem.foo = MDOFunction(lambda x: x, "foo")
    evaluation_problem._function_names.append("foo")
    evaluation_problem.stop_if_nan = False
    assert not evaluation_problem.stop_if_nan
    assert not evaluation_problem._stop_if_nan


@pytest.mark.parametrize("is_function_input_normalized", [False, True])
def test_jacobian_is_none_and_maxiter_is_reached(
    evaluation_problem, is_function_input_normalized
):
    """Check that an error is raised when Jacobian is None and maxiter is reached."""
    evaluation_problem.preprocess_functions(
        is_function_input_normalized=is_function_input_normalized
    )
    evaluation_problem.evaluation_counter.current = 1
    evaluation_problem.evaluation_counter.maximum = 1
    with pytest.raises(MaxIterReachedException):
        evaluation_problem.observables[0].jac(array([0.0]))


@pytest.mark.parametrize(
    ("jacobian_functions", "expected"),
    [
        ((Rosenbrock().objective,), {"rosen": array([-2.0, 0.0])}),
        ((), {"rosen": array([-2.0, 0.0])}),
        (None, {}),
    ],
)
def test_evaluate_jacobian_functions(jacobian_functions, expected):
    """Check the jacobian_functions argument of evaluate_functions."""
    data = Rosenbrock().evaluate_functions(jacobian_functions=jacobian_functions)[1]
    assert_equal(data, expected)
