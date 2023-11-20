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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the SciPy ODE solver wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import arange
from numpy import array
from numpy import exp
from numpy import sqrt
from numpy import sum
from numpy import zeros
from numpy.linalg import norm

from gemseo.algos.ode.lib_scipy_ode import ScipyODEAlgos
from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.algos.ode.ode_solvers_factory import ODESolversFactory
from gemseo.problems.ode.orbital_dynamics import OrbitalDynamics
from gemseo.problems.ode.van_der_pol import VanDerPol

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
def test_factory(algo_name):
    """Test the factory for ODE solvers."""
    assert ODESolversFactory().is_available(algo_name)


@parametrized_algo_names
def test_scipy_ode_algos(algo_name):
    """Test the wrapper for SciPy ODE solvers."""
    algos = ScipyODEAlgos()
    assert algo_name in algos.algorithms


@pytest.mark.parametrize("time_vector", [None, arange(0, 1, 0.1)])
def test_ode_problem_1d(time_vector):
    r"""Test the definition and resolution of an ODE problem.

    Define and solve the problem :math:`f'(t, s(t)) = s(t)` with the initial state
    :math:`f(0) = 1`. Compare the solution returned by the solver to the known
    analytical solution :math:`f: s \mapsto \exp(s)`. Root-mean-square of difference
    bewteen analytical and     approximated solutions should be small.
    """

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:  # noqa:U100
        return array(state)

    def _jac(time: float, state: NDArray[float]) -> NDArray[float]:  # noqa:U100
        return array(1)

    _initial_state = [1]
    _initial_time = 0
    _final_time = 1

    problem = ODEProblem(
        _func,
        jac=_jac,
        initial_state=_initial_state,
        initial_time=_initial_time,
        final_time=_final_time,
        time_vector=time_vector,
    )
    assert not problem.result.is_converged
    assert problem.result.n_func_evaluations == 0
    assert problem.result.n_jac_evaluations == 0
    assert problem.result.state_vector.size == 0
    assert problem.result.time_vector.size == 0

    algo_name = "DOP853"
    ODESolversFactory().execute(problem, algo_name, first_step=1e-6)

    analytical_solution = exp(problem.result.time_vector)
    assert sqrt(sum((problem.result.state_vector - analytical_solution) ** 2)) < 1e-6

    problem.check()

    assert problem.rhs_function == _func
    assert problem.jac == _jac
    assert len(problem.initial_state) == 1
    assert problem.initial_state == _initial_state
    assert problem.result.state_vector.size != 0
    assert problem.result.state_vector.size == problem.result.time_vector.size
    assert problem.result.solver_name == algo_name
    assert (
        problem.result.solver_message
        == "The solver successfully reached the end of the integration interval."
    )
    assert problem.result.is_converged
    assert problem.integration_interval == (_initial_time, _final_time)
    assert problem.result.n_func_evaluations > 0


def test_ode_problem_2d():
    r"""Test the definition and resolution of an ODE problem.

    Define and solve the problem :math:`f'(t, s(t)) = s(t)` with the initial state
    :math:`f(0, 0) = 1`. The jacobian of this problem is the identity matrix.
    """

    def _func(time: float, state: NDArray[float]) -> NDArray[float]:  # noqa:U100
        return state

    def _jac(time: float, state: NDArray[float]) -> NDArray[float]:  # noqa:U100
        return array([[1, 0], [0, 1]])

    problem = ODEProblem(
        _func,
        jac=_jac,
        initial_state=[1, 1],
        initial_time=0,
        final_time=1,
        time_vector=arange(0, 1, 0.1),
    )
    algo_name = "DOP853"
    ODESolversFactory().execute(problem, algo_name, first_step=1e-6)
    state_vect = [0.0, 1.0]
    problem.check_jacobian(state_vect)
    assert problem.result.is_converged
    assert problem.result.solver_name == algo_name
    assert problem.result.state_vector is not None

    analytical_solution = exp(problem.result.time_vector)
    assert sqrt(sum((problem.result.state_vector[0] - analytical_solution) ** 2)) < 1e-6


def test_ode_problem_2d_wrong_jacobian():
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
        jac=_jac,
        initial_state=[1, 1],
        initial_time=0,
        final_time=1,
        time_vector=arange(0, 1, 0.1),
    )
    algo_name = "DOP853"
    ODESolversFactory().execute(problem, algo_name, first_step=1e-6)
    state_vect = [0.0, 1.0]
    try:
        problem.check_jacobian(state_vect)
    except ValueError:
        pass
    else:
        raise ValueError("Jacobian should not be considered correct.")


@parametrized_algo_names
def test_van_der_pol(algo_name):
    """Solve Van der Pol with the jacobian analytical expression."""
    problem = VanDerPol()
    ODESolversFactory().execute(problem, algo_name, first_step=10e-6)
    assert problem.result.is_converged
    assert norm(problem.result.state_vector) > 0
    assert (
        problem.result.solver_message == "The solver successfully reached the "
        "end of the integration interval."
    )
    problem.check()


@parametrized_algo_names
def test_van_der_pol_finite_differences(algo_name):
    """Solve Van der Pol using finite differences for the jacobian."""
    problem = VanDerPol(use_jacobian=False)
    ODESolversFactory().execute(problem, algo_name, first_step=10e-6)
    assert problem.result.is_converged
    assert norm(problem.result.state_vector) > 0
    assert (
        problem.result.solver_message == "The solver successfully reached the "
        "end of the integration interval."
    )


def test_van_der_pol_jacobian_explicit_expression():
    """Validate the analytical expression of the jacobian."""
    problem = VanDerPol()
    state_vect = [0.0, 0.0]
    problem.check_jacobian(state_vect)
    assert not problem.result.is_converged


@pytest.mark.parametrize(
    ("algo_name", "eccentricity"),
    [
        ("RK45", 0.5),
        ("RK23", 0.5),
        ("DOP853", 0.5),
        ("Radau", 0.5),
        ("BDF", 0.5),
        ("LSODA", 0.5),
        ("RK45", 0),
        ("RK45", 0.1),
        ("RK45", 0.8),
    ],
)
@pytest.mark.parametrize("use_jacobian", [True, False])
def test_orbital(algo_name, eccentricity, use_jacobian):
    """Solve the orbital problem."""
    problem = OrbitalDynamics(eccentricity=eccentricity, use_jacobian=use_jacobian)
    ODESolversFactory().execute(problem, algo_name, first_step=10e-6)
    assert problem.result.is_converged


def test_orbital_jacobian_explicit_expression():
    """Validate the analytical expression of the jacobian."""
    problem = OrbitalDynamics()
    problem.time_vect = array([0.0, 1.0])
    state_vect = [0.0, 0.0, 0.0, 0.0]
    problem.check_jacobian(state_vect)
    assert not problem.result.is_converged


def test_unconverged(caplog):
    r"""Test solver behavior when it doesn't converge.

    Consider the equation :math:`s' = s^2` with the initial condition :math:`s(0) = 1`.
    This initial value problem should not converge on the interval :math:`[0, 1]`.
    """
    caplog.clear()

    algo_name = "RK45"

    def _func(time, state):
        return state**2

    problem = ODEProblem(_func, initial_state=[1], initial_time=0, final_time=1)
    ODESolversFactory().execute(problem, algo_name=algo_name, first_step=1e-6)

    assert not problem.result.is_converged
    assert f"The ODE solver {algo_name} did not converge." in caplog.records[1].message


@pytest.mark.parametrize("problem", [OrbitalDynamics, VanDerPol])
def test_check_ode_problem(problem):
    """Ensure the check method of ODEProblem behaves as expected."""
    problem = problem()
    assert problem.result.state_vector.size == 0
    problem.check()

    ODESolversFactory().execute(problem)
    assert problem.result.state_vector is not None
    problem.check()

    problem.result.time_vector = zeros(0)
    try:
        problem.check()
    except ValueError:
        pass
    else:
        raise ValueError
