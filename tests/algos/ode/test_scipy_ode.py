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
#        :author: Isabelle Santos
#        :author: Giulio Gargantini
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the SciPy ODE solver wrapper."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import arange
from numpy import arctan
from numpy import array
from numpy import cos
from numpy import exp
from numpy import isclose
from numpy import linspace
from numpy import sin
from numpy import sqrt
from numpy import sum as np_sum

from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.algos.ode.ode_problem import ODEResult
from gemseo.algos.ode.scipy_ode.scipy_ode import ScipyODEAlgos

if TYPE_CHECKING:
    from gemseo.typing import RealArray

parametrized_algo_names = pytest.mark.parametrize(
    "algo_name",
    [
        "RK45",
        "RK23",
        "DOP853",
        "Radau",
        "BDF",
        "LSODA",
    ],
)


@parametrized_algo_names
def test_factory(algo_name) -> None:
    """Test the factory for ODE solvers."""
    assert ODESolverLibraryFactory().is_available(algo_name)


@parametrized_algo_names
def test_scipy_ode_algos(algo_name) -> None:
    """Test the wrapper for SciPy ODE solvers."""
    assert algo_name in ScipyODEAlgos.ALGORITHM_INFOS


@pytest.mark.parametrize("times_eval", [None, arange(0, 1, 0.1)])
def test_ode_problem_1d(times_eval) -> None:
    r"""Test the definition and resolution of an ODE problem.

    Define and solve the problem :math:`f'(t, s(t)) = s(t)` with the initial state
    :math:`f(0) = 1`. Compare the solution returned by the solver to the known
    analytical solution :math:`f: s \mapsto \exp(s)`. Root-mean-square of difference
    bewteen analytical and     approximated solutions should be small.
    """

    def _func(time: float, state: RealArray) -> RealArray:  # noqa:U100
        return array(state)

    def _jac_wrt_state(time: float, state: RealArray) -> RealArray:  # noqa:U100
        return array(1)

    initial_state = array([1])
    initial_time = 0
    final_time = 1

    problem = ODEProblem(
        _func,
        jac_function_wrt_state=_jac_wrt_state,
        initial_state=initial_state,
        times=array([initial_time, final_time]),
        solve_at_algorithm_times=True,
    )
    assert not problem.result.algorithm_has_converged
    assert problem.result.n_func_evaluations == 0
    assert problem.result.n_jac_evaluations == 0
    assert problem.result.state_trajectories.size == 0
    assert problem.result.times.size == 0

    algo_name = "DOP853"
    assert isinstance(
        ODESolverLibraryFactory().execute(
            problem, algo_name=algo_name, first_step=1e-6
        ),
        ODEResult,
    )

    analytical_solution = exp(problem.result.times)
    difference = problem.result.state_trajectories - analytical_solution
    assert sqrt(np_sum(difference**2)) < 1e-6

    assert problem.rhs_function == _func
    assert problem.jac_function_wrt_state == _jac_wrt_state
    assert len(problem.initial_state) == 1
    assert problem.initial_state == initial_state
    assert problem.result.state_trajectories.size != 0
    assert problem.result.state_trajectories.size == problem.result.times.size
    assert problem.result.algorithm_name == algo_name
    assert (
        problem.result.algorithm_termination_message
        == "The solver successfully reached the end of the integration interval."
    )
    assert problem.result.algorithm_has_converged
    assert problem.time_interval == (initial_time, final_time)
    assert problem.result.n_func_evaluations > 0


def test_ode_problem_2d() -> None:
    r"""Test the definition and resolution of an ODE problem.

    Define and solve the problem :math:`f'(t, s(t)) = s(t)` with the initial state
    :math:`f(0, 0) = 1`. The jacobian of this problem is the identity matrix.
    """

    def func(time: float, state: RealArray) -> RealArray:  # noqa:U100
        return state

    def jac_wrt_state(time: float, state: RealArray) -> RealArray:  # noqa:U100
        return array([[1, 0], [0, 1]])

    problem = ODEProblem(
        func,
        jac_function_wrt_state=jac_wrt_state,
        initial_state=array([1, 1]),
        times=arange(0, 1, 0.1),
    )
    problem.check_jacobian(array([0.0, 1.0]))
    algo_name = "DOP853"
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=1e-6)
    assert problem.result.algorithm_has_converged
    assert problem.result.algorithm_name == algo_name
    assert problem.result.state_trajectories is not None

    analytical_solution = exp(problem.result.times)
    assert (
        sqrt(sum((problem.result.state_trajectories[0] - analytical_solution) ** 2))
        < 1e-6
    )


def test_ode_problem_2d_array_jacobian() -> None:
    r"""Test the definition and resolution of an ODE problem.

    Define and solve the problem :math:`f'(t, s(t)) = s(t)` with the initial state
    :math:`f(0, 0) = 1`. The jacobian of this problem is the identity matrix.
    """

    def func(time: float, state: RealArray) -> RealArray:  # noqa:U100
        return state

    problem = ODEProblem(
        func,
        jac_function_wrt_state=array([[1, 0], [0, 1]]),
        initial_state=array([1, 1]),
        times=arange(0, 1, 0.1),
    )

    problem.check_jacobian(array([0.0, 1.0]))
    algo_name = "DOP853"
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=1e-6)
    assert problem.result.algorithm_has_converged
    assert problem.result.algorithm_name == algo_name
    assert problem.result.state_trajectories is not None

    analytical_solution = exp(problem.result.times)
    assert (
        sqrt(sum((problem.result.state_trajectories[0] - analytical_solution) ** 2))
        < 1e-6
    )


def test_ode_problem_2d_array_time_state_callable_jacobian() -> None:
    r"""Test the definition and resolution of an ODE problem.

    Define and solve the problem :math:`f(t, s(t)) = cos^2 (s(t))` with the initial
    state :math:`f(0, 0) = 1`.
    The Jacobian of this problem with respect to time and state is:

    .. math:: Jac_t[f](t, s) = [0, 2 \sin(2 s)]`
        .
    """

    def func(time: float, state: RealArray) -> RealArray:  # noqa:U100
        return cos(state) ** 2

    def jacobian_wrt_state(time: float, state: RealArray) -> RealArray:
        return (-sin(2 * state)).reshape((1, -1))

    problem = ODEProblem(
        func,
        jac_function_wrt_state=jacobian_wrt_state,
        initial_state=array([0]),
        times=arange(0, 1, 0.1),
    )

    problem.check_jacobian(array([1.0]), error_max=1e-6)
    algo_name = "DOP853"
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=1e-6)
    assert problem.result.algorithm_has_converged
    assert problem.result.algorithm_name == algo_name
    assert problem.result.state_trajectories is not None

    analytical_solution = arctan(problem.result.times)
    assert allclose(problem.result.state_trajectories, analytical_solution, atol=1e-5)


def test_ode_problem_2d_wrong_jacobian() -> None:
    r"""Test that check_jacobian raises an error when the jacobian is wrong.

    Define and solve the problem :math:`f'(t, s(t)) = s(t)` with the initial state
    :math:`f(0, 0) = 1`.
    """

    def _func(time, state):  # noqa:U100
        return array(state)

    def _jac(time, state):  # noqa:U100
        return array([[1.1, 0.0], [0.0, 1.0]])

    problem = ODEProblem(
        _func,
        jac_function_wrt_state=_jac,
        initial_state=array([1, 1]),
        times=arange(0, 1, 0.1),
    )
    algo_name = "DOP853"
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=1e-6)
    try:
        problem.check_jacobian(array([0.0, 1.0]))
    except ValueError:
        pass
    else:
        msg = "Jacobian should not be considered correct."
        raise ValueError(msg)


def test_ode_problem_without_jacobian() -> None:
    r"""Test that check_jacobian raises an error when the jacobian is not given, but it
    asked to be tested."""

    def _func(time, state):  # noqa:U100
        return array(state)

    problem = ODEProblem(
        _func,
        initial_state=array([1, 1]),
        times=arange(0, 1, 0.1),
    )
    algo_name = "DOP853"
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=1e-6)

    with pytest.raises(
        AttributeError,
        match=re.escape("The function jac_function_wrt_state is not available."),
    ):
        problem.check_jacobian(array([1.0]))


def test_unconverged(caplog) -> None:
    r"""Test solver behavior when it doesn't converge.

    Consider the equation :math:`s' = s^2` with the initial condition :math:`s(0) = 1`.
    This initial value problem should not converge on the interval :math:`[0, 1]`.
    """
    caplog.clear()

    algo_name = "RK45"

    def _func(time, state):
        return state**2

    problem = ODEProblem(_func, array([1]), array([0.0, 1.0]))
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=1e-6)

    assert not problem.result.algorithm_has_converged
    assert f"The ODE solver {algo_name} did not converge." in caplog.records[1].message


def test_problem_without_given_time_interval():
    r"""Test for when the time interval is not provided.

    Consider the equation :math:`s' = s`, the initial condition :math:`s(0) = 1`. This
    initial value problem should not converge on the interval :math:`[0, 1]`.
    """

    algo_name = "RK45"

    array([1])
    time_interval = array([0.0, 1.0])

    def _func(time, state):
        return state

    problem = ODEProblem(
        func=_func,
        initial_state=array([1]),
        times=time_interval,
        solve_at_algorithm_times=True,
    )
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=1e-6)

    reference_sol = exp(problem.result.times)
    assert problem.result.algorithm_has_converged
    assert allclose(problem.result.state_trajectories, reference_sol, atol=1e-3)


def test_inconsistent_space_and_time_shapes():
    algo_name = "RK45"

    def _func(time, state):
        return state

    problem = ODEProblem(_func, initial_state=array([1]), times=linspace(0.0, 0.5, 10))
    times_2 = linspace(0.0, 0.5, 3)
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=1e-6)

    problem.result.times = times_2

    with pytest.raises(ValueError) as error_info:
        problem.check()

    msg = "Inconsistent state and time shapes."
    assert msg in str(error_info.value)


def test_terminating_event() -> None:
    algo_name = "RK45"
    gravity_acceleration = -9.81
    initial_height = 10
    t_max = 4
    times = linspace(0.0, t_max, 30)

    def _func(time, state):
        return array([state[1], gravity_acceleration])

    def terminating_impact(time, state):
        return state[0]

    def exact_solution(times):
        return initial_height + times * times * gravity_acceleration / 2

    jac_wrt_state = array([[0, 1], [0, 0]])

    problem = ODEProblem(
        _func,
        initial_state=array([initial_height, 0]),
        times=times,
        jac_function_wrt_state=jac_wrt_state,
        event_functions=(terminating_impact,),
    )

    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, atol=1e-8)

    reference_sol = exact_solution(problem.result.times)
    assert allclose(reference_sol, problem.result.state_trajectories[0, :], atol=1e-6)
    assert isclose(problem.result.final_state[0], 0.0, atol=1e-3)


def test_terminating_event_fixed_times() -> None:
    algo_name = "RK45"
    gravity_acceleration = -9.81
    initial_height = 10
    t_max = 4
    times = linspace(0.0, t_max, 30)

    def _func(time, state):
        return array([state[1], gravity_acceleration])

    def terminating_impact(time: float, state: RealArray) -> RealArray:
        return state[0]

    def exact_solution(times):
        return initial_height + times * times * gravity_acceleration / 2

    jac_wrt_state = array([[0, 1], [0, 0]])

    problem = ODEProblem(
        _func,
        initial_state=array([initial_height, 0]),
        times=times,
        jac_function_wrt_state=jac_wrt_state,
        event_functions=(terminating_impact,),
        solve_at_algorithm_times=False,
    )

    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, atol=1e-8)

    reference_sol = exact_solution(problem.result.times)
    impact_instant = sqrt(-2 * initial_height / gravity_acceleration)
    assert allclose(reference_sol, problem.result.state_trajectories[0, :], atol=1e-6)
    assert isclose(impact_instant, problem.result.termination_time, atol=1e-6)


def test_terminating_event_outside_time_interval() -> None:
    algo_name = "RK45"
    gravity_acceleration = -9.81
    initial_height = 500
    t_max = 4
    times = linspace(0.0, t_max, 30)

    def _func(time, state):
        return array([state[1], gravity_acceleration])

    def terminating_impact(time, state):
        return state[0]

    def exact_solution(times):
        return initial_height + times * times * gravity_acceleration / 2

    jac_wrt_state = array([[0, 1], [0, 0]])

    problem = ODEProblem(
        _func,
        initial_state=array([initial_height, 0]),
        times=times,
        jac_function_wrt_state=jac_wrt_state,
        event_functions=(terminating_impact,),
        solve_at_algorithm_times=False,
    )

    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, atol=1e-8)

    reference_sol = exact_solution(problem.result.times)
    assert allclose(reference_sol, problem.result.state_trajectories[0, :], atol=1e-6)
    assert problem.result.terminal_event_index is None


def test_multiple_terminating_events() -> None:
    algo_name = "RK45"
    gravity_acceleration = -9.81
    initial_height = 10
    t_max = 4
    times = linspace(0.0, t_max, 30)

    def _func(time, state):
        return array([state[1], gravity_acceleration])

    def terminating_impact_floor(time, state):
        return state[0]

    def terminating_impact_ceiling(time, state):
        return 20.0 - state[0]

    def exact_solution(times):
        return initial_height + times * times * gravity_acceleration / 2

    jac_wrt_state = array([[0, 1], [0, 0]])

    problem_1 = ODEProblem(
        _func,
        initial_state=array([initial_height, 0]),
        times=times,
        jac_function_wrt_state=jac_wrt_state,
        event_functions=(terminating_impact_floor, terminating_impact_ceiling),
    )

    problem_2 = ODEProblem(
        _func,
        initial_state=array([initial_height, 0]),
        times=times,
        jac_function_wrt_state=jac_wrt_state,
        event_functions=(terminating_impact_ceiling, terminating_impact_floor),
    )

    ODESolverLibraryFactory().execute(problem_1, algo_name=algo_name, atol=1e-8)
    ODESolverLibraryFactory().execute(problem_2, algo_name=algo_name, atol=1e-8)

    reference_sol = exact_solution(problem_1.result.times)

    assert allclose(reference_sol, problem_1.result.state_trajectories[0, :], atol=1e-6)
    assert allclose(reference_sol, problem_2.result.state_trajectories[0, :], atol=1e-6)
    assert isclose(problem_1.result.final_state[0], 0.0, atol=1e-3)
    assert isclose(problem_2.result.final_state[0], 0.0, atol=1e-3)


def test_order_initial_and_final_times():
    initial_height = 10
    t_max = 4
    new_t_max = -1
    times = linspace(0.0, t_max, 30)

    def _func(time, state):
        return array([state[1], 1.0])

    problem = ODEProblem(
        _func,
        initial_state=array([initial_height, 0]),
        times=times,
    )

    msg = "The initial time must be lower than the final time."
    with pytest.raises(ValueError, match=re.escape(msg)):
        problem.update_times(final_time=new_t_max)
