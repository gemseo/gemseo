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

from functools import partial
from math import sqrt
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.lib_custom import CustomDOE
from gemseo.algos.doe.lib_pydoe import PyDOE
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.algos.stop_criteria import DesvarIsNan
from gemseo.algos.stop_criteria import FunctionIsNan
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.api import execute_algo
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.problems.sobieski.disciplines import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiStructure
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
from scipy.linalg import norm
from scipy.optimize import rosen
from scipy.optimize import rosen_der

DIRNAME = Path(__file__).parent
FAIL_HDF = DIRNAME / "fail2.hdf5"


@pytest.fixture(scope="module")
def problem_executed_twice() -> OptimizationProblem:
    """A problem executed twice."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "obj")
    problem.add_observable(MDOFunction(lambda x: x, "obs"))
    problem.add_constraint(MDOFunction(lambda x: x, "cstr", f_type="ineq"))

    execute_algo(problem, "CustomDOE", algo_type="doe", samples=array([[0.0]]))
    problem.current_iter = 0
    execute_algo(problem, "CustomDOE", algo_type="doe", samples=array([[0.5]]))
    return problem


@pytest.fixture
def pow2_problem() -> OptimizationProblem:
    design_space = DesignSpace()
    design_space.add_variable("x", 3, l_b=-1.0, u_b=1.0)
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
        args=["x"],
    )
    return problem


def test_init():
    design_space = DesignSpace()
    OptimizationProblem(design_space)


def test_checks():
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


def test_callback():
    """Test the execution of a callback."""
    n = 3
    design_space = DesignSpace()
    design_space.add_variable("x", n, l_b=-1.0, u_b=1.0)
    design_space.set_current_value(np.zeros(n))
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(rosen, name="rosen", f_type="obj", jac=rosen_der)
    problem.check()

    call_me = mock.Mock()
    problem.add_callback(call_me)
    problem.preprocess_functions()
    problem.check()

    problem.objective(problem.design_space.get_current_value())
    call_me.assert_called_once()


def test_add_constraints(pow2_problem):
    problem = pow2_problem
    ineq1 = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq1",
        f_type="ineq",
        jac=Power2.ineq_constraint1_jac,
        expr="0.5 -x[0] ** 3",
        args=["x"],
    )
    problem.add_ineq_constraint(ineq1, value=-1)
    assert problem.get_ineq_constraints_number() == 1
    assert problem.get_eq_constraints_number() == 0

    problem.add_ineq_constraint(ineq1, value=-1)
    problem.add_ineq_constraint(ineq1, value=-1)

    assert problem.get_constraints_number() == 3
    assert problem.has_nonlinear_constraints()

    ineq2 = MDOFunction(Power2.ineq_constraint1, name="ineq2")
    with pytest.raises(ValueError):
        problem.add_constraint(ineq2)

    problem.add_constraint(ineq1, positive=True)

    problem.constraints = [problem.objective]
    with pytest.raises(ValueError):
        problem.check()


def test_getmsg_ineq_constraints(pow2_problem):
    expected = []
    problem = pow2_problem

    ineq_std = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_std",
        f_type="ineq",
        expr="cstr + cst",
        args=["x"],
    )
    problem.add_constraint(ineq_std)
    expected.append("ineq_std(x): cstr + cst <= 0.0")

    ineq_lo_posval = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_lo_posval",
        f_type="ineq",
        expr="cstr + cst",
        args=["x"],
    )
    problem.add_constraint(ineq_lo_posval, value=1.0)
    expected.append("ineq_lo_posval(x): cstr + cst <= 1.0")

    ineq_lo_negval = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_lo_negval",
        f_type="ineq",
        expr="cstr + cst",
        args=["x"],
    )
    problem.add_constraint(ineq_lo_negval, value=-1.0)
    expected.append("ineq_lo_negval(x): cstr + cst <= -1.0")

    ineq_up_negval = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_up_negval",
        f_type="ineq",
        expr="cstr + cst",
        args=["x"],
    )
    problem.add_constraint(ineq_up_negval, value=-1.0, positive=True)
    expected.append("ineq_up_negval(x): cstr + cst >= -1.0")

    ineq_up_posval = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq_up_posval",
        f_type="ineq",
        expr="cstr + cst",
        args=["x"],
    )
    problem.add_constraint(ineq_up_posval, value=1.0, positive=True)
    expected.append("ineq_up_posval(x): cstr + cst >= 1.0")

    linear_constraint = MDOLinearFunction(array([1, 2]), "lin1", f_type="ineq")
    problem.add_constraint(linear_constraint)
    expected.append("lin1(x!0, x!1): x!0 + 2.00e+00*x!1 <= 0.0")

    linear_constraint = MDOLinearFunction(array([1, 2]), "lin2", f_type="ineq")
    problem.add_constraint(linear_constraint, positive=True, value=-1.0)
    expected.append("lin2(x!0, x!1): x!0 + 2.00e+00*x!1 >= -1.0")

    msg = str(problem)
    for elem in expected:
        assert elem in msg


def test_getmsg_eq_constraints(pow2_problem):
    expected = []
    problem = pow2_problem

    eq_std = MDOFunction(
        Power2.ineq_constraint1,
        name="eq_std",
        f_type="eq",
        expr="cstr + cst",
        args=["x"],
    )
    problem.add_constraint(eq_std)
    expected.append("eq_std(x): cstr + cst == 0.0")

    eq_posval = MDOFunction(
        Power2.ineq_constraint1,
        name="eq_posval",
        f_type="eq",
        expr="cstr + cst",
        args=["x"],
    )
    problem.add_constraint(eq_posval, value=1.0)
    expected.append("eq_posval(x): cstr + cst == 1.0")
    eq_negval = MDOFunction(
        Power2.ineq_constraint1,
        name="eq_negval",
        f_type="eq",
        expr="cstr + cst",
        args=["x"],
    )
    problem.add_constraint(eq_negval, value=-1.0)
    expected.append("eq_negval(x): cstr + cst == -1.0")

    msg = str(problem)
    for elem in expected:
        assert elem in msg


def test_get_dimension(pow2_problem):
    problem = pow2_problem
    problem.u_bounds = None
    problem.l_bounds = None
    dim = 3
    assert problem.get_dimension() == dim
    problem.u_bounds = np.ones(3)
    assert problem.get_dimension() == dim
    problem.l_bounds = -np.ones(3)
    assert problem.get_dimension() == dim


def test_check_format(pow2_problem):
    problem = pow2_problem
    with pytest.raises(TypeError):
        problem.check_format("1")


def test_constraints_dim(pow2_problem):
    problem = pow2_problem
    ineq1 = MDOFunction(
        Power2.ineq_constraint1,
        name="ineq1",
        f_type="ineq",
        jac=Power2.ineq_constraint1_jac,
        expr="0.5 -x[0] ** 3",
        args=["x"],
    )
    problem.add_ineq_constraint(ineq1, value=-1)
    with pytest.raises(ValueError):
        problem.get_ineq_cstr_total_dim()
    assert problem.get_eq_constraints_number() == 0
    assert len(problem.get_nonproc_constraints()) == 0
    problem.preprocess_functions()
    assert len(problem.get_nonproc_constraints()) == 1


def test_check():
    # Objective is missing!
    design_space = DesignSpace()
    design_space.add_variable("x", 3, l_b=-1.0, u_b=1.0)
    design_space.set_current_value(np.array([1.0, 1.0, 1.0]))
    problem = OptimizationProblem(design_space)
    with pytest.raises(ValueError):
        problem.check()


def test_missing_constjac(pow2_problem):
    problem = pow2_problem
    ineq1 = MDOFunction(sum, name="sum", f_type="ineq", expr="sum(x)", args=["x"])
    problem.add_ineq_constraint(ineq1, value=-1)
    problem.preprocess_functions()
    with pytest.raises(ValueError):
        problem.evaluate_functions(ones(3), eval_jac=True)


def _test_check_bounds(pow2_problem):
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


def test_pb_type(pow2_problem):
    problem = pow2_problem
    problem.pb_type = "None"
    with pytest.raises(TypeError):
        problem.check()


def test_differentiation_method_without_current_x(pow2_problem):
    """Check that a ValueError is raised when the current x is not defined."""
    pow2_problem.design_space.set_current_value({})
    with pytest.raises(ValueError, match=r"The design space has no current value\."):
        pow2_problem._OptimizationProblem__add_fd_jac("foo", False)


def test_differentiation_method(pow2_problem):
    problem = pow2_problem
    problem.differentiation_method = problem.COMPLEX_STEP
    problem.fd_step = 0.0
    with pytest.raises(ValueError):
        problem.check()
    problem.fd_step = 1e-7 + 1j * 1.0e-7
    problem.check()
    problem.fd_step = 1j * 1.0e-7
    problem.check()

    problem.differentiation_method = problem.FINITE_DIFFERENCES
    problem.fd_step = 0.0
    with pytest.raises(ValueError):
        problem.check()
    problem.fd_step = 1e-7 + 1j * 1.0e-7
    problem.check()


def test_get_dv_names():
    problem = Power2()
    OptimizersFactory().execute(problem, "SLSQP")
    assert problem.design_space.variables_names == ["x"]


def test_get_best_infeasible_point():
    problem = Power2()
    x_opt, f_opt, is_opt_feasible, _ = problem.get_best_infeasible_point()
    assert x_opt is None
    assert f_opt is None
    assert not is_opt_feasible
    problem.preprocess_functions()
    x_0 = problem.design_space.normalize_vect(zeros(3))
    f_val = problem.objective(x_0)
    x_opt, f_opt, is_opt_feasible, opt_fd = problem.get_best_infeasible_point()
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
    x_opt, f_opt, is_opt_feasible, opt_fd = problem.get_best_infeasible_point()
    assert not is_opt_feasible
    assert x_opt is not None
    assert f_opt is not None
    assert len(opt_fd) > 0


def test_feasible_optimum_points():
    problem = Power2()
    with pytest.raises(ValueError):
        problem.get_optimum()

    OptimizersFactory().execute(
        problem, "SLSQP", eq_tolerance=1e-6, ineq_tolerance=1e-6
    )
    feasible_points, _ = problem.get_feasible_points()
    assert len(feasible_points) >= 2
    min_value, solution, is_feasible, _, _ = problem.get_optimum()
    assert (solution == feasible_points[-1]).all()
    assert allclose(min_value, 2.192090802, 9)
    assert allclose(solution[0], 0.79370053, 8)
    assert allclose(solution[1], 0.79370053, 8)
    assert allclose(solution[2], 0.96548938, 8)
    assert is_feasible


def test_nan():
    problem = Power2()
    problem.preprocess_functions()

    with pytest.raises(DesvarIsNan):
        problem.objective(array([1.0, float("nan")]))

    with pytest.raises(FunctionIsNan):
        problem.objective.jac(array([1.0, float("nan")]))

    problem = Power2()
    problem.objective.jac = lambda x: array([float("nan")] * 3)
    problem.preprocess_functions()
    with pytest.raises(FunctionIsNan):
        problem.objective.jac(array([0.1, 0.2, 0.3]))


def test_preprocess_functions():
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
    assert id(problem.nonproc_objective) == obj_id
    assert {id(cstr) for cstr in problem.nonproc_constraints} == cstr_id
    assert {id(obs) for obs in problem.nonproc_observables} == obs_id

    # Check that the current problem functions are NOT the original ones
    assert id(problem.objective) != obj_id
    assert {id(cstr) for cstr in problem.constraints}.isdisjoint(cstr_id)
    assert {id(obs) for obs in problem.observables}.isdisjoint(obs_id)

    nonproc_constraints = {repr(cstr) for cstr in problem.nonproc_constraints}
    constraints = {repr(cstr) for cstr in problem.constraints}
    assert nonproc_constraints == constraints


def test_normalize_linear_function():
    """Test the normalization of linear functions."""
    design_space = DesignSpace()
    lower_bounds = array([-5.0, -7.0])
    upper_bounds = array([11.0, 13.0])
    x_0 = 0.2 * lower_bounds + 0.8 * upper_bounds
    design_space.add_variable("x", 2, l_b=lower_bounds, u_b=upper_bounds, value=x_0)
    objective = MDOLinearFunction(
        array([[2.0, 0.0], [0.0, 3.0]]), "affine", "obj", "x", array([5.0, 7.0])
    )
    low_bnd_value = objective(lower_bounds)
    upp_bnd_value = objective(upper_bounds)
    initial_value = objective(x_0)
    problem = OptimizationProblem(design_space)
    problem.objective = objective
    problem.preprocess_functions(use_database=False, round_ints=False)
    assert allclose(problem.objective(zeros(2)), low_bnd_value)
    assert allclose(problem.objective(ones(2)), upp_bnd_value)
    assert allclose(problem.objective(0.8 * ones(2)), initial_value)


def test_export_hdf(tmp_wd):
    file_path = Path("power2.h5")
    problem = Power2()
    OptimizersFactory().execute(problem, "SLSQP")
    problem.export_hdf(file_path, append=True)  # Shall still work now

    def check_pb(imp_pb):
        assert file_path.exists()
        assert str(imp_pb) == str(problem)
        assert str(imp_pb.solution) == str(problem.solution)
        assert file_path.exists()

        assert problem.get_eq_cstr_total_dim() == 1
        assert problem.get_ineq_cstr_total_dim() == 2

    problem.export_hdf(file_path)

    imp_pb = OptimizationProblem.import_hdf(file_path)
    check_pb(imp_pb)

    problem.export_hdf(file_path, append=True)
    imp_pb = OptimizationProblem.import_hdf(file_path)
    check_pb(imp_pb)
    val = imp_pb.objective(imp_pb.database.get_x_by_iter(1))
    assert isinstance(val, float)
    jac = imp_pb.objective.jac(imp_pb.database.get_x_by_iter(0))
    assert isinstance(jac, ndarray)
    with pytest.raises(ValueError):
        imp_pb.objective(array([1.1254]))


def test_evaluate_functions():
    """Evaluate the functions of the Power2 problem."""
    problem = Power2()
    func, grad = problem.evaluate_functions(
        x_vect=array([1.0, 0.5, 0.2]),
        eval_jac=True,
        eval_obj=False,
        normalize=False,
    )
    assert "pow2" not in func
    assert "pow2" not in grad
    assert func["ineq1"] == pytest.approx(array([-0.5]))
    assert func["ineq2"] == pytest.approx(array([0.375]))
    assert func["eq"] == pytest.approx(array([0.892]))
    assert grad["ineq1"] == pytest.approx(array([-3.0, 0.0, 0.0]))
    assert grad["ineq2"] == pytest.approx(array([0.0, -0.75, 0.0]))
    assert grad["eq"] == pytest.approx(array([0.0, 0.0, -0.12]))


def test_evaluate_functions_no_gradient():
    """Evaluate the functions of the Power2 problem without computing the gradients."""
    problem = Power2()

    func, grad = problem.evaluate_functions(
        normalize=False, no_db_no_norm=True, eval_obj=False
    )
    assert "pow2" not in func
    assert "pow2" not in grad
    assert func["ineq1"] == pytest.approx(array([-0.5]))
    assert func["ineq2"] == pytest.approx(array([-0.5]))
    assert func["eq"] == pytest.approx(array([-0.1]))


def test_evaluate_functions_only_gradients():
    """Evaluate the gradients of the Power2 problem without evaluating the functions."""
    func, grad = Power2().evaluate_functions(
        normalize=False,
        no_db_no_norm=True,
        eval_obj=False,
        constraints_names=[],
        jacobians_names=["ineq1", "ineq2", "eq"],
    )
    assert not func
    assert grad.keys() == {"ineq1", "ineq2", "eq"}
    assert grad["ineq1"] == pytest.approx(array([-3, 0, 0]))
    assert grad["ineq2"] == pytest.approx(array([0, -3, 0]))
    assert grad["eq"] == pytest.approx(array([0, 0, -3]))


@pytest.mark.parametrize("no_db_no_norm", [True, False])
def test_evaluate_functions_w_observables(pow2_problem, no_db_no_norm):
    """Test the evaluation of the fuctions of a problem with observables."""
    problem = pow2_problem
    design_norm = "design norm"
    observable = MDOFunction(norm, design_norm)
    problem.add_observable(observable)
    problem.preprocess_functions()
    out = problem.evaluate_functions(
        x_vect=array([1.0, 1.0, 1.0]),
        normalize=False,
        no_db_no_norm=no_db_no_norm,
        eval_observables=True,
    )
    assert out[0]["pow2"] == pytest.approx(3.0)
    assert out[0]["design norm"] == pytest.approx(sqrt(3.0))


def test_evaluate_functions_non_preprocessed(constrained_problem):
    """Check the evaluation of non-preprocessed functions."""
    values, jacobians = constrained_problem.evaluate_functions(
        normalize=False, no_db_no_norm=True
    )
    assert set(values.keys()) == {"f", "g", "h"}
    assert values["f"] == pytest.approx(2.0)
    assert values["g"] == pytest.approx(array([1.0]))
    assert values["h"] == pytest.approx(array([1.0, 1.0]))
    assert jacobians == dict()


@pytest.mark.parametrize(
    ["pre_normalize", "eval_normalize", "x_vect"],
    [
        (False, False, array([0.1, 0.2, 0.3])),
        (False, True, array([0.55, 0.6, 0.65])),
        (True, False, array([0.1, 0.2, 0.3])),
        (True, True, array([0.55, 0.6, 0.65])),
    ],
)
def test_evaluate_functions_preprocessed(pre_normalize, eval_normalize, x_vect):
    """Check the evaluation of preprocessed functions."""
    constrained_problem = Power2()
    constrained_problem.preprocess_functions(is_function_input_normalized=pre_normalize)
    values, _ = constrained_problem.evaluate_functions(
        x_vect=x_vect, normalize=eval_normalize
    )
    assert set(values.keys()) == {"pow2", "ineq1", "ineq2", "eq"}
    assert values["pow2"] == pytest.approx(0.14)
    assert values["ineq1"] == pytest.approx(array([0.499]))
    assert values["ineq2"] == pytest.approx(array([0.492]))
    assert values["eq"] == pytest.approx(array([0.873]))


@pytest.mark.parametrize("preprocess_functions", [False, True])
@pytest.mark.parametrize("no_db_no_norm", [False, True])
@pytest.mark.parametrize(
    ["constraints_names", "keys"], [[None, {"g", "h"}], [["h"], {"h"}]]
)
def test_evaluate_constraints_subset(
    constrained_problem, preprocess_functions, no_db_no_norm, constraints_names, keys
):
    """Check the evaluation of a subset of constraints."""
    if preprocess_functions:
        constrained_problem.preprocess_functions()

    values, _ = constrained_problem.evaluate_functions(
        array([0, 0]),
        eval_obj=False,
        no_db_no_norm=no_db_no_norm,
        constraints_names=constraints_names,
    )
    assert values.keys() == keys


@pytest.mark.parametrize(
    ["observables_names", "eval_observables", "keys"],
    [
        [None, False, {"f", "g", "h"}],
        [None, True, {"f", "g", "h", "a", "b"}],
        [["a"], False, {"f", "g", "h", "a"}],
        [["a"], True, {"f", "g", "h", "a"}],
    ],
)
def test_evaluate_observables_subset(
    constrained_problem, observables_names, eval_observables, keys
):
    """Check the evaluation of a subset of observables."""
    values, _ = constrained_problem.evaluate_functions(
        array([0, 0]),
        eval_observables=eval_observables,
        observables_names=observables_names,
    )
    assert values.keys() == keys


@pytest.mark.parametrize(
    ["jacobians_names", "eval_jac", "keys"],
    [
        [None, False, set()],
        [None, True, {"f", "g", "h"}],
        [["h", "b"], False, {"h", "b"}],
        [["h", "b"], True, {"h", "b"}],
    ],
)
def test_evaluate_jacobians_subset(
    constrained_problem, jacobians_names, eval_jac, keys
):
    """Check the evaluation of the Jacobian matrices for a subset of the functions."""
    _, jacobians = constrained_problem.evaluate_functions(
        array([0, 0]), eval_jac, jacobians_names=jacobians_names
    )
    assert jacobians.keys() == keys


@pytest.mark.parametrize(
    ["jacobian_names", "message"],
    [[["unknown"], "This name is"], [["other unknown", "unknown"], "These names are"]],
)
def test_evaluate_unknown_jacobians(constrained_problem, jacobian_names, message):
    """Check the evaluation of the Jacobian matrices of unknown functions."""
    with pytest.raises(
        ValueError,
        match=f"{message} not among the names of the functions: {', '.join(jacobian_names)}.",
    ):
        constrained_problem.evaluate_functions(
            array([0, 0]), jacobians_names=jacobian_names
        )


@pytest.mark.parametrize("eval_jac", [False, True])
@pytest.mark.parametrize(
    ["jacobians_names", "keys"], [[["h"], {"h"}], [list(), set()], [None, set()]]
)
def test_evaluate_jacobians_alone(constrained_problem, eval_jac, jacobians_names, keys):
    """Check the evaluation of Jacobian matrices alone."""
    values, jacobians = constrained_problem.evaluate_functions(
        array([0, 0]),
        eval_jac,
        False,
        constraints_names=[],
        jacobians_names=jacobians_names,
    )
    assert not values
    assert jacobians.keys() == keys


def test_no_normalization():
    problem = Power2()
    OptimizersFactory().execute(problem, "SLSQP", normalize_design_space=False)
    f_opt, _, is_feas, _, _ = problem.get_optimum()
    assert is_feas
    assert abs(f_opt - 2.192) < 0.01


def test_nan_func():
    problem = Power2()

    def nan_func(_):
        return float("nan")

    problem.objective.func = nan_func
    problem.preprocess_functions()
    with pytest.raises(FunctionIsNan):
        problem.objective(zeros(3))


def test_fail_import():
    with pytest.raises(KeyError):
        OptimizationProblem.import_hdf(FAIL_HDF)


def test_append_export(tmp_wd):
    """Test the export of an HDF5 file with append mode.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    problem = Rosenbrock()
    problem.preprocess_functions()
    func = problem.objective
    file_path_db = "test_pb_append.hdf5"
    # Export empty file
    problem.export_hdf(file_path_db)

    n_calls = 200
    for i in range(n_calls):
        func(array([0.1, 1.0 / (i + 1.0)]))

    # Export again with append mode
    problem.export_hdf(file_path_db, append=True)

    read_db = Database(file_path_db)
    assert len(read_db) == n_calls

    i += 1
    func(array([0.1, 1.0 / (i + 1.0)]))

    # Export again with identical elements plus a new one.
    problem.export_hdf(file_path_db, append=True)
    read_db = Database(file_path_db)
    assert len(read_db) == n_calls + 1


def test_grad_normalization(pow2_problem):
    problem = pow2_problem
    x_vec = ones(3)
    grad = problem.objective.jac(x_vec)
    problem.preprocess_functions()
    norm_grad = problem.objective.jac(x_vec)

    assert 0.0 == pytest.approx(norm(norm_grad - 2 * grad))

    unnorm_grad = problem.design_space.normalize_vect(norm_grad, minus_lb=False)
    assert 0.0 == pytest.approx(norm(unnorm_grad - grad))


def test_2d_objective():
    disc = SobieskiStructure()
    design_space = SobieskiProblem().design_space
    inputs = disc.get_input_data_names()
    design_space.filter([name for name in inputs if not name.startswith("c_")])
    doe_scenario = DOEScenario([disc], "DisciplinaryOpt", "y_12", design_space)
    doe_scenario.execute({"algo": "DiagonalDOE", "n_samples": 10})


def test_observable(pow2_problem):
    """Test the handling of observables.

    Args:
        pow2_problem: The Power2 problem.
    """
    problem = pow2_problem
    design_norm = "design norm"
    observable = MDOFunction(norm, design_norm)
    problem.add_observable(observable)

    # Check that the observable can be found
    assert problem.get_observable(design_norm) is observable
    with pytest.raises(ValueError):
        problem.get_observable("toto")

    # Check that the observable is stored in the database
    OptimizersFactory().execute(problem, "SLSQP")
    database = problem.database
    iter_norms = [norm(key.unwrap()) for key in database.keys()]
    iter_obs = [value[design_norm] for value in database.values()]
    assert iter_obs == iter_norms

    # Check that the observable is exported
    dataset = problem.export_to_dataset("dataset")
    func_data = dataset.get_data_by_group("functions", as_dict=True)
    obs_data = func_data.get(design_norm)
    assert obs_data is not None
    assert func_data[design_norm][:, 0].tolist() == iter_norms
    assert dataset.GRADIENT_GROUP not in dataset.groups
    dataset = problem.export_to_dataset("dataset", export_gradients=True)
    assert dataset.GRADIENT_GROUP in dataset.groups
    name = Database.get_gradient_name("pow2")
    n_iter = len(database)
    n_var = problem.design_space.dimension
    assert dataset.get_data_by_names(name, as_dict=False).shape == (n_iter, n_var)


@pytest.mark.parametrize(
    "filter_non_feasible,as_dict,expected",
    [
        (
            True,
            True,
            {
                "x": array(
                    [[1.0, 1.0, np.power(0.9, 1 / 3)], [0.9, 0.9, np.power(0.9, 1 / 3)]]
                )
            },
        ),
        (
            True,
            False,
            np.array(
                [[1.0, 1.0, np.power(0.9, 1 / 3)], [0.9, 0.9, np.power(0.9, 1 / 3)]]
            ),
        ),
        (
            False,
            True,
            {
                "x": array(
                    [
                        [1.0, 1.0, np.power(0.9, 1 / 3)],
                        [0.9, 0.9, np.power(0.9, 1 / 3)],
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5],
                    ]
                )
            },
        ),
        (
            False,
            False,
            np.array(
                [
                    [1.0, 1.0, np.power(0.9, 1 / 3)],
                    [0.9, 0.9, np.power(0.9, 1 / 3)],
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                ]
            ),
        ),
    ],
)
def test_get_data_by_names(filter_non_feasible, as_dict, expected):
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
    data = problem.get_data_by_names(
        names=["x"], as_dict=as_dict, filter_non_feasible=filter_non_feasible
    )
    # Check output is filtered when needed
    if as_dict:
        assert np.array_equal(data["x"], expected["x"])
    else:
        assert np.array_equal(data, expected)


def test_gradient_with_random_variables():
    """Check that the Jacobian is correctly computed with random variable."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("x", "OTUniformDistribution")

    problem = OptimizationProblem(parameter_space)
    problem.objective = MDOFunction(lambda x: 3 * x**2, "func", jac=lambda x: 6 * x)
    PyDOE().execute(problem, "fullfact", n_samples=3, eval_jac=True)

    data = problem.database.get_func_grad_history("func")

    assert array_equal(data, array([0.0, 3.0, 6.0]))


def test_is_mono_objective():
    """Check the boolean OptimizationProblem.is_mono_objective."""
    design_space = DesignSpace()
    design_space.add_variable("")
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(
        lambda x: array([1.0, 2.0]), name="func", f_type="obj", outvars=["y1", "y2"]
    )

    assert not problem.is_mono_objective

    problem.objective = MDOFunction(
        lambda x: x, name="func", f_type="obj", outvars=["y1"]
    )

    assert problem.is_mono_objective


def test_undefined_differentiation_method():
    """Check that passing an undefined differentiation raises a ValueError."""
    with pytest.raises(
        ValueError,
        match=(
            "'foo' is not a differentiation methods; "
            "available ones are: "
            "'user', 'complex_step', 'finite_differences', 'no_derivatives'."
        ),
    ):
        OptimizationProblem(DesignSpace(), differentiation_method="foo")


@pytest.fixture
def problem() -> OptimizationProblem:
    """A simple optimization problem :math:`max_x x`."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0, u_b=1, value=0.5)
    opt_problem = OptimizationProblem(design_space)
    opt_problem.objective = MDOFunction(lambda x: x, name="func", f_type="obj")
    return opt_problem


def test_parallel_differentiation(problem):
    """Check that parallel_differentiation is taken into account."""
    assert not problem.parallel_differentiation
    problem.parallel_differentiation = True
    assert problem.parallel_differentiation


def test_parallel_differentiation_options(problem):
    """Check that parallel_differentiation_options is taken into account."""
    assert not problem.parallel_differentiation_options
    problem.parallel_differentiation_options = {"step": 1e-10}
    assert problem.parallel_differentiation_options == {"step": 1e-10}


def test_parallel_differentiation_setting_after_functions_preprocessing(problem):
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


def test_database_name(problem):
    """Check the name of the database."""
    DOEFactory().execute(problem, "fullfact", n_samples=1)
    problem.database.name = "my_database"
    dataset = problem.export_to_dataset()
    assert dataset.name == problem.database.name
    dataset = problem.export_to_dataset("dataset")
    assert dataset.name == "dataset"


@pytest.mark.parametrize(
    "skip_int_check,expected_message",
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
def test_int_opt_problem(skip_int_check, expected_message, caplog):
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
    design_space.add_variable("x", l_b=1, u_b=3, value=array([1]), var_type="integer")
    problem = OptimizationProblem(design_space)
    problem.objective = -f_1

    if skip_int_check:
        OptimizersFactory().execute(
            problem,
            "SLSQP",
            normalize_design_space=True,
            skip_int_check=skip_int_check,
        )
        assert expected_message in caplog.text
        assert problem.get_optimum()[1] == array([2.0])
    else:
        with pytest.raises(ValueError, match=expected_message):
            OptimizersFactory().execute(
                problem,
                "SLSQP",
                normalize_design_space=True,
                skip_int_check=skip_int_check,
            )


@pytest.fixture(scope="function")
def constrained_problem() -> OptimizationProblem:
    """A constrained optimisation problem with multidimensional constraints."""
    design_space = DesignSpace()
    design_space.add_variable("x", 2, value=1.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x.sum(), "f", jac=lambda x: [1, 1])
    problem.add_constraint(
        MDOFunction(lambda x: x[0], "g", jac=lambda _: [1, 0]), cstr_type="ineq"
    )
    problem.add_constraint(
        MDOFunction(lambda x: x, "h", jac=lambda _: np.eye(2)), cstr_type="eq"
    )
    problem.add_observable(MDOFunction(lambda x: x[1], "a", jac=lambda x: [0, 1]))
    problem.add_observable(MDOFunction(lambda x: -x, "b", jac=lambda x: -np.eye(2)))
    return problem


@pytest.mark.parametrize(
    ["names", "dimensions"],
    [(None, {"f": 1, "g": 1, "h": 2}), (["g", "h"], {"g": 1, "h": 2})],
)
def test_get_functions_dimensions(constrained_problem, names, dimensions):
    """Check the computation of the functions dimensions."""
    assert constrained_problem.get_functions_dimensions(names) == dimensions


@pytest.mark.parametrize(
    ["design", "n_unsatisfied"],
    [
        (array([0.0, 0.0]), 0),
        (array([-1.0, 0.0]), 1),
        (array([-1.0, -1.0]), 2),
        (array([1.0, 1.0]), 3),
    ],
)
def test_get_number_of_unsatisfied_constraints(
    constrained_problem, design, n_unsatisfied
):
    """Check the computation of the number of unsatisfied constraints."""
    assert (
        constrained_problem.get_number_of_unsatisfied_constraints(design)
        == n_unsatisfied
    )


def test_get_scalar_constraints_names(constrained_problem):
    """Check the computation of the scalar constraints names."""
    scalar_names = constrained_problem.get_scalar_constraints_names()
    assert set(scalar_names) == {
        "g",
        f"h{DesignSpace.SEP}0",
        f"h{DesignSpace.SEP}1",
    }


def test_observables_callback():
    """Test that the observables are called properly."""
    problem = Power2()
    obs1 = MDOFunction(norm, "design_norm")
    problem.add_observable(obs1)
    problem.database.store(
        array([0.79499653, 0.20792012, 0.96630481]),
        {"pow2": 1.61, "ineq1": -0.0024533, "ineq2": -0.0024533, "eq": -0.00228228},
    )

    problem.preprocess_functions(is_function_input_normalized=False)
    problem.execute_observables_callback(
        last_x=array([0.79499653, 0.20792012, 0.96630481])
    )

    assert obs1.n_calls == 1


def test_approximated_jacobian_wrt_uncertain_variables():
    """Check that the approximated Jacobian wrt uncertain variables is correct."""
    uspace = ParameterSpace()
    uspace.add_random_variable("u", "OTNormalDistribution")
    problem = OptimizationProblem(uspace)
    problem.differentiation_method = problem.FINITE_DIFFERENCES
    problem.objective = MDOFunction(lambda u: u, "func")
    CustomDOE().execute(problem, "CustomDOE", samples=array([[0.0]]), eval_jac=True)
    grad = problem.database.get_func_grad_history("func")
    assert grad[0, 0] == pytest.approx(1.0, abs=1e-3)


@pytest.fixture
def rosenbrock_lhs() -> tuple[Rosenbrock, dict[str, ndarray]]:
    """The Rosenbrock problem after evaluation and its start point."""
    problem = Rosenbrock()
    problem.add_observable(MDOFunction(lambda x: sum(x), "obs"))
    problem.add_constraint(MDOFunction(lambda x: sum(x), "cstr"), cstr_type="ineq")
    start_point = problem.design_space.get_current_value(as_dict=True)
    execute_algo(problem, "lhs", n_samples=3, algo_type="doe")
    return problem, start_point


def test_reset(rosenbrock_lhs):
    """Check the default behavior of OptimizationProblem.reset."""
    problem, start_point = rosenbrock_lhs
    nonproc_functions = (
        [problem.nonproc_objective]
        + problem.nonproc_constraints
        + problem.nonproc_observables
        + problem.nonproc_new_iter_observables
    )
    problem.reset()
    assert len(problem.database) == 0
    assert problem.database.get_max_iteration() == 0
    assert not problem._OptimizationProblem__functions_are_preprocessed
    for key, val in problem.design_space.get_current_value(as_dict=True).items():
        assert (start_point[key] == val).all()

    functions = (
        [problem.objective]
        + problem.constraints
        + problem.observables
        + problem.new_iter_observables
    )
    for func, nonproc_func in zip(functions, nonproc_functions):
        assert id(func) == id(nonproc_func)

    assert problem.nonproc_objective is None
    assert problem.nonproc_constraints == []
    assert problem.nonproc_observables == []
    assert problem.nonproc_new_iter_observables == []


def test_reset_database(rosenbrock_lhs):
    """Check OptimizationProblem.reset without database reset."""
    problem, start_point = rosenbrock_lhs
    problem.reset(database=False)
    assert len(problem.database) == 3
    assert problem.database.get_max_iteration() == 3


def test_reset_current_iter(rosenbrock_lhs):
    """Check OptimizationProblem.reset without current_iter reset."""
    problem, start_point = rosenbrock_lhs
    problem.reset(current_iter=False)
    assert len(problem.database) == 0
    assert problem.database.get_max_iteration() == 3


def test_reset_design_space(rosenbrock_lhs):
    """Check OptimizationProblem.reset without design_space reset."""
    problem, start_point = rosenbrock_lhs
    problem.reset(design_space=False)
    for key, val in problem.design_space.get_current_value(as_dict=True).items():
        assert (start_point[key] != val).any()


def test_reset_functions(rosenbrock_lhs):
    """Check OptimizationProblem.reset without reset the number of function calls."""
    problem, start_point = rosenbrock_lhs
    problem.reset(function_calls=False)
    assert problem.objective.n_calls == 3


def test_reset_wo_current_value():
    """Check OptimizationProblem.reset when the default design value is missing."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "obj")
    problem.design_space.set_current_value({"x": array([0.0])})
    problem.reset()
    assert problem.design_space.get_current_value(as_dict=True) == {}


def test_reset_preprocess(rosenbrock_lhs):
    """Check OptimizationProblem.reset without functions pre-processing reset."""
    problem, start_point = rosenbrock_lhs
    problem.reset(preprocessing=False)
    assert problem._OptimizationProblem__functions_are_preprocessed
    functions = (
        [problem.objective]
        + problem.constraints
        + problem.observables
        + problem.new_iter_observables
    )
    nonproc_functions = (
        [problem.nonproc_objective]
        + problem.nonproc_constraints
        + problem.nonproc_observables
        + problem.nonproc_new_iter_observables
    )
    for func, nonproc_func in zip(functions, nonproc_functions):
        assert id(func) != id(nonproc_func)

    assert problem.nonproc_objective is not None
    assert len(problem.nonproc_constraints) == len(problem.constraints)
    assert len(problem.nonproc_observables) == len(problem.observables)
    assert len(problem.nonproc_new_iter_observables) == len(
        problem.new_iter_observables
    )


def test_function_string_representation_from_hdf():
    """Check the string representation of a function when importing a HDF5 file.

    The commented code is the one used for creating the HDF5 file.
    """
    # design_space = DesignSpace()
    # design_space.add_variable("x0", l_b=0.0, u_b=1.0, value=0.5)
    # design_space.add_variable("x1", l_b=0.0, u_b=1.0, value=0.5)
    # problem = OptimizationProblem(design_space)
    # problem.objective = MDOFunction(lambda x: x[0] + x[1], "f", args=["x0", "x1"])
    # problem.constraints.append(
    #     MDOFunction(lambda x: x[0] + x[1], "g", args=["x0", "x1"])
    # )
    # problem.export_hdf("opt_problem_to_check_string_representation.hdf5")

    new_problem = OptimizationProblem.import_hdf(
        DIRNAME / "opt_problem_to_check_string_representation.hdf5"
    )
    assert str(new_problem.objective) == "f(x0, x1)"
    assert str(new_problem.constraints[0]) == "g(x0, x1)"


@pytest.mark.parametrize(["name", "dimension"], [("f", 1), ("g", 1), ("h", 2)])
def test_get_function_dimension(constrained_problem, name, dimension):
    """Check the output dimension of a problem function."""
    assert constrained_problem.get_function_dimension(name) == dimension


def test_get_function_dimension_unknown(constrained_problem):
    """Check the output dimension of an unknown problem function."""
    with pytest.raises(ValueError, match="The problem has no function named unknown."):
        constrained_problem.get_function_dimension("unknown")


@pytest.fixture()
def design_space() -> mock.Mock:
    """A design space."""
    design_space = mock.Mock()
    design_space.get_current_x = mock.Mock()
    return design_space


@pytest.fixture()
def function() -> mock.Mock:
    """A function."""
    function = mock.MagicMock(return_value=1.0)
    function.name = "f"
    return function


@pytest.mark.parametrize(["expects_normalized"], [(True,), (False,)])
def test_get_function_dimension_no_dim(function, design_space, expects_normalized):
    """Check the implicitly defined output dimension of a problem function."""
    function.has_dim = mock.Mock(return_value=False)
    function.expects_normalized_inputs = expects_normalized
    design_space.has_current_value = mock.Mock(return_value=True)
    problem = OptimizationProblem(design_space)
    problem.objective = function
    problem.get_x0_normalized = mock.Mock()
    assert problem.get_function_dimension(function.name) == 1
    if expects_normalized:
        problem.get_x0_normalized.assert_called_once()
        design_space.get_current_value.assert_called_once()
    else:
        problem.get_x0_normalized.assert_not_called()
        assert design_space.get_current_value.call_count == 2


def test_get_function_dimension_unavailable(function, design_space):
    """Check the unavailable output dimension of a problem function."""
    function.has_dim = mock.Mock(return_value=False)
    design_space.has_current_value = mock.Mock(return_value=False)
    problem = OptimizationProblem(design_space)
    problem.objective = function
    with pytest.raises(
        RuntimeError,
        match="The output dimension of function {} is not available.".format(
            function.name
        ),
    ):
        problem.get_function_dimension(function.name)


@pytest.mark.parametrize("categorize", [True, False])
@pytest.mark.parametrize("export_gradients", [True, False])
def test_dataset_missing_values(categorize, export_gradients):
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
            "Iter": [1],
            "design norm": 1.7320508075688772,
            "@pow2": array([2.0, 2.0, 2.0]),
        },
    )
    # Add a point with missing values.
    problem.database.store(np.array([-1.0, -1.0, -1.0]), {"Iter": [2]})
    # Add a complete evaluation.
    problem.database.store(
        np.array([-1.77635684e-15, 1.33226763e-15, 4.44089210e-16]),
        {
            "pow2": 5.127595883936577e-30,
            "Iter": [3],
            "design norm": 2.2644195468014703e-15,
            "@pow2": array([-3.55271368e-15, 2.66453526e-15, 8.88178420e-16]),
        },
    )
    # Add one evaluation with complete function data but missing gradient.
    problem.database.store(
        np.array([0.0, 0.0, 0.0]), {"pow2": 0.0, "Iter": [4], "design norm": 0.0}
    )
    # Another point with missing values.
    problem.database.store(
        np.array([0.5, 0.5, 0.5]),
        {"Iter": [5]},
    )
    # Export to a dataset.
    dataset = problem.export_to_dataset(
        categorize=categorize, export_gradients=export_gradients
    )
    # Check that the missing values are exported as NaN.
    if categorize:
        if export_gradients:
            assert dataset.data["functions"][3].all() == np.array([0.0, 0.0]).all()
            assert (
                dataset.data["gradients"][3].all()
                == np.array([np.nan, np.nan, np.nan]).all()
            )
        else:
            assert (
                dataset.data["functions"][1].all() == np.array([np.nan, np.nan]).all()
            )

    else:
        if export_gradients:
            assert (
                dataset.data["parameters"][3].all()
                == np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.nan, np.nan, np.nan]).all()
            )
        else:
            assert (
                dataset.data["parameters"][4].all()
                == np.array([0.5, 0.5, 0.5, np.nan, np.nan]).all()
            )


@pytest.fixture
def problem_for_eval_obs_jac() -> OptimizationProblem:
    """An optimization problem to check the option eval_obs_jac."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.differentiation_method = problem.FINITE_DIFFERENCES
    problem.objective = MDOFunction(lambda x: x, "f", jac=lambda x: 1)
    problem.add_constraint(
        MDOFunction(lambda x: x, "c", f_type="ineq", jac=lambda x: 1)
    )
    problem.add_observable(MDOFunction(lambda x: x, "o", jac=lambda x: 1))
    return problem


@pytest.mark.parametrize(
    "options",
    [
        {"algo_name": "SLSQP", "algo_type": "opt", "max_iter": 1},
        {"algo_name": "fullfact", "algo_type": "doe", "n_samples": 1},
    ],
)
@pytest.mark.parametrize("eval_obs_jac", [True, False])
def test_observable_jac(problem_for_eval_obs_jac, options, eval_obs_jac):
    """Check that the observable derivatives are computed when eval_obs_jac is True."""
    execute_algo(problem_for_eval_obs_jac, eval_obs_jac=eval_obs_jac, **options)
    assert problem_for_eval_obs_jac.database.contains_dataname("@o") is eval_obs_jac


def test_presence_observables_hdf_file(pow2_problem, tmp_wd):
    """Check if the observables can be retrieved in an HDF file after export and
    import."""
    # Add observables to the optimization problem.
    obs1 = MDOFunction(norm, "design norm")
    pow2_problem.add_observable(obs1)
    obs2 = MDOFunction(lambda x: sum(x), "sum")
    pow2_problem.add_observable(obs2)

    OptimizersFactory().execute(pow2_problem, "SLSQP")

    # Export and import the optimization problem.
    file_path = "power2.h5"
    pow2_problem.export_hdf(file_path)
    imp_pb = OptimizationProblem.import_hdf(file_path)

    # Check the set of observables.
    # Assuming that two functions are equal if they have the same name.
    exp_obs_names = {obs.name for obs in pow2_problem.observables}
    imp_obs_names = {obs.name for obs in imp_pb.observables}
    assert exp_obs_names == imp_obs_names


@pytest.mark.parametrize(
    "input_values,expected",
    [
        (None, array([[1.0], [2.0]])),
        (array([[1.0], [2.0], [1.0]]), array([[1.0], [2.0], [1.0]])),
    ],
)
def test_export_to_dataset(input_values, expected):
    """Check the export of the database."""
    design_space = DesignSpace()
    design_space.add_variable("dv")

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x * 2, "obj")
    problem.constraints.append(
        MDOFunction(lambda x: x * 3, "cstr", f_type=MDOFunction.TYPE_INEQ)
    )

    algo = CustomDOE()
    algo.algo_name = "CustomDOE"
    algo.execute(problem, samples=array([[1.0], [2.0], [1.0]]))

    dataset = problem.export_to_dataset(input_values=input_values, by_group=False)
    assert_equal(
        dataset.data,
        {
            "dv": expected,
            "obj": expected * 2,
            "cstr": expected * 3,
        },
    )


@pytest.mark.parametrize("name", ["a", "c"])
def test_export_to_dataset_input_names_order(name):
    """Check that the order of the input names is not changed in the dataset."""
    design_space = DesignSpace()
    design_space.add_variable("b")
    design_space.add_variable(name)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x[0] + x[1], "obj")

    algo = CustomDOE()
    algo.algo_name = "CustomDOE"
    algo.execute(problem, samples=array([[1.0, 1.0], [2.0, 2.0]]))

    dataset = problem.export_to_dataset()
    assert dataset.get_names("design_parameters") == ["b", name]


@pytest.fixture(scope="module")
def problem_with_complex_value() -> OptimizationProblem:
    """A problem using a design space with a float variable whose value is complex."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.set_current_value({"x": array([1.0 + 0j])})
    return OptimizationProblem(design_space)


@pytest.mark.parametrize("cast", [False, True])
def test_get_x0_normalized_no_complex(problem_with_complex_value, cast):
    """Check that the complex value of a float variable is converted to float."""
    normalized_x0 = problem_with_complex_value.get_x0_normalized(cast_to_real=cast)
    assert (normalized_x0.dtype.kind == "c") is not cast


def test_objective_name():
    """Check the name of the objective."""
    problem = OptimizationProblem(DesignSpace())
    problem.objective = MDOFunction(lambda x: x, "f")
    assert problem.get_objective_name() == "f"
    assert problem.get_objective_name(False) == "f"
    problem.change_objective_sign()
    assert problem.get_objective_name() == "-f"
    assert problem.get_objective_name(False) == "f"


@pytest.mark.parametrize("cstr_type", [MDOFunction.TYPE_EQ, MDOFunction.TYPE_INEQ])
@pytest.mark.parametrize("has_default_name", [False, True])
@pytest.mark.parametrize(
    "value,positive,name",
    [
        (None, False, "c"),
        (None, True, "-c"),
        (1.0, True, "-c + 1.0"),
        (-1.0, True, "-c - 1.0"),
        (1.0, False, "c - 1.0"),
        (-1.0, False, "c + 1.0"),
    ],
)
def test_constraint_name(has_default_name, value, positive, cstr_type, name):
    """Check the name of a constraint."""
    problem = OptimizationProblem(DesignSpace())
    constraint_function = MDOFunction(lambda x: x, "c")
    constraint_function.has_default_name = has_default_name
    problem.add_constraint(
        constraint_function, value=value, positive=positive, cstr_type=cstr_type
    )
    cstr_name = problem.constraints[0].name
    if not has_default_name:
        assert cstr_name == "c"
    else:
        assert cstr_name == name

    assert problem.constraint_names["c"] == [cstr_name]


def test_observables_normalization():
    """Test that the observables are called at each iteration."""
    disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
    design_space = DesignSpace()
    design_space.add_variable("x_local", l_b=0.0, u_b=10.0, value=ones(1))
    design_space.add_variable(
        "x_shared", 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=array([4.0, 3.0])
    )
    scenario = create_scenario(
        disciplines, formulation="MDF", objective_name="obj", design_space=design_space
    )
    scenario.add_constraint("c_1", "ineq")
    scenario.add_constraint("c_2", "ineq")
    scenario.add_observable("y_1")
    scenario.execute(input_data={"max_iter": 3, "algo": "SLSQP"})
    total_iter = scenario.formulation.opt_problem.database.get_max_iteration()
    n_obj_eval = scenario.formulation.opt_problem.database.get_func_history("y_1").size
    n_obs_eval = scenario.formulation.opt_problem.database.get_func_history("obj").size
    assert total_iter == n_obj_eval == n_obs_eval


def test_observable_cannot_be_added_twice(caplog):
    """Check that an observable cannot be added twice."""
    problem = OptimizationProblem(DesignSpace())
    problem.add_observable(MDOFunction(lambda x: x, "obs"))
    problem.add_observable(MDOFunction(lambda x: x, "obs"))
    assert "WARNING" in caplog.text
    assert 'The optimization problem already observes "obs".' in caplog.text
    assert len(problem.observables) == 1


def test_repr_constraint_linear_lower_ineq():
    """Check the representation of a linear lower inequality-constraint."""
    design_space = DesignSpace()
    design_space.add_variable("x", 2)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOLinearFunction(array([1, 2]), "f")
    problem.add_ineq_constraint(
        MDOLinearFunction(
            array([[0, 1], [2, 3], [4, 5]]), "g", value_at_zero=array([6, 7, 8])
        ),
        positive=True,
    )
    assert str(problem) == (
        """Optimization problem:
   minimize f(x!0, x!1) = x!0 + 2.00e+00*x!1
   with respect to x
   subject to constraints:
      g(x!0, x!1): [ 0.00e+00  1.00e+00][x!0] + [ 6.00e+00] >= 0.0
                   [ 2.00e+00  3.00e+00][x!1]   [ 7.00e+00]
                   [ 4.00e+00  5.00e+00]        [ 8.00e+00]"""
    )


def test_get_original_observable(pow2_problem):
    """Check the accessor to an original observable."""
    function = MDOFunction(None, "f")
    pow2_problem.add_observable(function)
    assert pow2_problem.get_observable(function.name) is function


def test_get_preprocessed_observable(pow2_problem):
    """Check the accessor to a pre-processed observable."""
    function = MDOFunction(None, "f")
    pow2_problem.add_observable(function)
    pow2_problem.preprocess_functions()
    assert pow2_problem.get_observable(function.name) is pow2_problem.observables[-1]


def test_get_missing_observable(constrained_problem):
    """Check the accessor to a missing observable."""
    with pytest.raises(
        ValueError,
        match="missing_observable_name is not among the names of the observables: a, b.",
    ):
        constrained_problem.get_observable("missing_observable_name")


@pytest.mark.parametrize("name", ["obj", "cstr", "obj"])
def test_execute_twice(problem_executed_twice, name):
    """Check that the second evaluations of an OptimizationProblem works."""
    assert len(problem_executed_twice.database.get_func_history(name)) == 2


def test_avoid_complex_in_dataset():
    """Check that exporting database to dataset casts complex numbers to real."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(
        lambda x: array([0j]), "f", jac=lambda x: array([[0j]])
    )
    problem.preprocess_functions()
    problem.evaluate_functions(array([0.25 + 0j]), eval_jac=True)
    dataset = problem.export_to_dataset(export_gradients=True)
    for name in ["@f", "f", "x"]:
        assert dataset[name].dtype.kind == "f"


@pytest.mark.parametrize("cstr_type", ["eq", "ineq"])
def test_nan_get_violation_criteria(cstr_type):
    """Test get_violation_criteria in the presence of NaN constraints."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.ineq_tolerance = 1e-3
    problem.eq_tolerance = 1e-3
    problem.objective = MDOFunction(lambda x: x, "obj")
    problem.add_constraint(MDOFunction(lambda x: x, "cstr"), cstr_type=cstr_type)

    x_vect1 = array([1.0])
    problem.database.store(x_vect1, {"obj": array([1]), "cstr": array([float("NaN")])})

    is_pt_feasible, f_violation = problem.get_violation_criteria(x_vect1)
    assert not is_pt_feasible
    assert f_violation == float("inf")

    x_vect2 = array([2.0])
    problem.database.store(x_vect2, {"obj": array([1.0]), "cstr": array([0.0])})
    is_pt_feasible2, f_violation2 = problem.get_violation_criteria(x_vect2)
    assert is_pt_feasible2
    assert f_violation2 == 0.0

    x_vect3 = array([3.0])
    problem.database.store(array([3.0]), {"obj": array([0.0]), "cstr": array([2.0])})
    is_pt_feasible3, f_violation3 = problem.get_violation_criteria(x_vect3)
    assert not is_pt_feasible3
    assert f_violation3 == (2.0 - 1e-3) ** 2

    opt = problem.get_optimum()
    assert opt[0] == 1.0
