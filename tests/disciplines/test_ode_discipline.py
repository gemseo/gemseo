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
"""Tests for disciplines based on ODEs.

ODE stands for Ordinary Differential Equation.
"""

from __future__ import annotations

import re
from math import atan
from math import pi

import pytest
from numpy import allclose
from numpy import arctan
from numpy import array
from numpy import concatenate
from numpy import cos
from numpy import exp
from numpy import isclose
from numpy import linspace
from numpy import sin
from numpy import sqrt
from numpy.testing import assert_allclose

from gemseo import from_pickle
from gemseo import to_pickle
from gemseo.core.discipline.base_discipline import CacheType
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.problems.ode.oscillator_discipline import OscillatorDiscipline


def test_create_oscillator_ode_discipline() -> None:
    """Test the creation of an ODE Discipline."""
    times = linspace(0.0, 10, 30)
    ode_disc = OscillatorDiscipline(times=times, omega=4)
    assert ode_disc is not None


def test_oscillator_ode_discipline_final_time() -> None:
    """Test an ODE Discipline representing a simple oscillator.

    The only part of the solution taken into account is the solution at the final time
    """

    times = (0.0, 10.0)
    oscillator_discipline = OscillatorDiscipline(omega=2, times=times)
    assert oscillator_discipline is not None

    final_time = times[-1]
    out = oscillator_discipline.execute()
    final_analytical_position = sin(2 * final_time) / 2
    assert allclose(out["final_position"], final_analytical_position)
    final_analytical_velocity = cos(2 * final_time)
    assert allclose(out["final_velocity"], final_analytical_velocity)


def test_oscillator_final_time_set_names() -> None:
    """Test an ODE Discipline representing a simple oscillator.

    The only part of the solution taken into account is the solution at the final time
    """

    times = linspace(0.0, 10, 30)
    final_state_names = {
        "position": "pos_at_final_time",
        "velocity": "vel_at_final_time",
    }
    oscillator_discipline = OscillatorDiscipline(
        omega=2, times=times, final_state_names=final_state_names
    )
    assert oscillator_discipline is not None

    final_time = times[-1]
    out = oscillator_discipline.execute()
    final_analytical_position = sin(2 * final_time) / 2
    assert allclose(out["pos_at_final_time"], final_analytical_position)
    final_analytical_velocity = cos(2 * final_time)
    assert allclose(out["vel_at_final_time"], final_analytical_velocity)


def test_oscillator_ode_discipline_trajectory() -> None:
    """Test an ODE Discipline representing a simple oscillator."""
    times = linspace(0.0, 10, 30)
    oscillator_discipline = OscillatorDiscipline(
        omega=2, times=times, return_trajectories=True
    )
    assert oscillator_discipline is not None

    out = oscillator_discipline.execute()
    analytical_position = sin(2 * times) / 2
    assert allclose(out["position"], analytical_position)
    analytical_velocity = cos(2 * times)
    assert allclose(out["velocity"], analytical_velocity)


def test_oscillator_different_initial_condition() -> None:
    """Test an ODE Discipline representing a simple oscillator."""
    times = linspace(0.0, 10, 30)
    omega = 2
    oscillator_discipline = OscillatorDiscipline(
        omega=omega, times=times, return_trajectories=True
    )
    assert oscillator_discipline is not None

    initial_position = array([3.0])
    initial_velocity = array([-2.0])

    amplitude = sqrt(initial_position**2 + (initial_velocity / omega) ** 2)
    if allclose(initial_velocity, [0.0]):
        phase = pi / 2
    else:
        phase = arctan(omega * initial_position / initial_velocity)

    if phase < 0.0:
        phase += pi

    out = oscillator_discipline.execute({
        "initial_position": initial_position,
        "initial_velocity": initial_velocity,
    })
    analytical_position = sin(omega * times + phase) * amplitude
    analytical_velocity = cos(omega * times + phase) * omega * amplitude
    assert allclose(out["position"], analytical_position)
    assert allclose(out["velocity"], analytical_velocity)


def test_ode_discipline_design_variable() -> None:
    """Test an ODEDiscipline with a design variable."""
    initial_time = array([0.0])
    initial_state = array([0.0])
    parameter = array([1.0])
    times = linspace(0.0, 10, 30)

    def dynamics(time=initial_time, state=initial_state, parameter=parameter):
        state_dot = parameter
        return state_dot  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=dynamics)
    discipline.set_cache(cache_type=CacheType.NONE)

    ode_discipline = ODEDiscipline(
        rhs_discipline=discipline,
        times=times,
        time_name="time",
        state_names={"state": "state_dot"},
        final_state_names={"state": "final_state"},
    )

    first_solution = ode_discipline.execute()
    assert isclose(first_solution["final_state"], 10.0)

    second_solution = ode_discipline.execute({"parameter": array([2.0])})
    assert isclose(second_solution["final_state"], 20.0)


def test_ode_discipline_time_dependence() -> None:
    initial_x = array([0.0])
    times = array([0.0, 1.0])

    def _fct(time=times[0], x=initial_x):
        x_dot = time
        return x_dot  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=_fct)
    discipline.set_cache(cache_type=CacheType.NONE)
    ode_discipline = ODEDiscipline(
        rhs_discipline=discipline,
        times=times,
        time_name="time",
        state_names={"x": "x_dot"},
        final_state_names={"x": "final_x"},
    )

    def exact_solution(t, init_x):
        return init_x + 0.5 * (t[1] ** 2 - t[0] ** 2)

    first_solution = ode_discipline.execute()
    first_exact_sol = exact_solution(times, initial_x)
    assert isclose(first_solution["final_x"], first_exact_sol)

    new_times = array([-3.0, 4.0])
    second_solution = ode_discipline.execute({
        "initial_time": new_times[0],
        "final_time": new_times[1],
    })
    second_exact_sol = exact_solution(new_times, initial_x)
    assert isclose(second_solution["final_x"], second_exact_sol)


def test_ode_trajectory_discipline_time_dependence() -> None:
    initial_x = array([0.0])
    times = linspace(0.0, 1.0, 20)

    def _fct(time=times[0], x=initial_x):
        x_dot = time
        return x_dot  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=_fct)
    discipline.set_cache(cache_type=CacheType.NONE)
    ode_discipline = ODEDiscipline(
        rhs_discipline=discipline,
        times=times,
        time_name="time",
        state_names={"x": "x_dot"},
        state_trajectory_names={"x": "trajectory_x"},
        return_trajectories=True,
    )

    def exact_solution(t, init_x):
        return init_x + 0.5 * (t**2 - t[0] ** 2)

    first_solution = ode_discipline.execute()
    first_exact_sol = exact_solution(times, initial_x)
    assert allclose(first_solution["trajectory_x"], first_exact_sol)

    new_times = linspace(-3.0, 4.0, 50)
    second_solution = ode_discipline.execute({"times": new_times})
    second_exact_sol = exact_solution(new_times, initial_x)
    assert allclose(second_solution["trajectory_x"], second_exact_sol)


def test_ode_discipline_with_design_variable() -> None:
    """Test an ODEDiscipline with a design variable."""
    initial_time = array([0.0])
    initial_state = array([0.0])
    parameter = array([1.0])
    times = linspace(0.0, 10, 30)

    def dynamics(time=initial_time, state=initial_state, parameter=parameter):
        state_dot = parameter
        return state_dot  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=dynamics)

    ode_discipline = ODEDiscipline(
        rhs_discipline=discipline,
        times=times,
        time_name="time",
        state_names={"state": "state_dot"},
        final_state_names={"state": "state_final"},
    )

    first_solution = ode_discipline.execute()
    assert isclose(first_solution["state_final"], 10.0)

    second_solution = ode_discipline.execute({"parameter": array([2.0])})
    assert isclose(second_solution["state_final"], 20.0)


def test_ode_discipline_bad_grammar() -> None:
    """Test error messages when passing an ill-formed grammar."""
    initial_time = array([0.0])
    initial_position = array([0.0])
    initial_velocity = array([1.0])

    def _rhs_function(
        time=initial_time,
        position=initial_position,
        velocity=initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = -4 * position
        return position_dot, velocity_dot

    oscillator = AutoPyDiscipline(py_func=_rhs_function)
    oscillator.set_cache(cache_type=CacheType.NONE)
    msg = (
        "'not_position' and 'not_velocity' are not input variables of "
        "the RHS discipline."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        bad_input_ode_discipline = ODEDiscipline(  # noqa: F841
            rhs_discipline=oscillator,
            state_names=["not_position", "not_velocity"],
            times=linspace(0.0, 10, 30),
        )


def test_ode_discipline_default_state_names() -> None:
    """Test the assignment of default names to the state variables in ODEDiscipline."""
    initial_time = array([0.0])
    initial_position = array([0.0])
    initial_velocity = array([1.0])

    def _rhs_function(
        time=initial_time,
        position=initial_position,
        velocity=initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = -4 * position
        return position_dot, velocity_dot

    oscillator = AutoPyDiscipline(py_func=_rhs_function)
    oscillator.set_cache(cache_type=CacheType.NONE)

    ode_discipline = ODEDiscipline(  # noqa: F841
        rhs_discipline=oscillator,
        times=linspace(0.0, 10, 30),
    )

    input_keys = ode_discipline.input_grammar.names
    assert (key in input_keys for key in ("time", "position", "velocity"))


def test_ode_discipline_wrong_ordering_time_derivatives():
    """Test the explicit ordering of the state derivatives in time."""
    times = array([0.0, 1.0])

    init_time = array([0.0])
    init_state_a = array([0.0])
    init_state_b = array([0.0])

    def _fct(time=init_time, a=init_state_a, b=init_state_b):
        a_dot = 1.0
        b_dot = -1.0
        return b_dot, a_dot  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=_fct)
    discipline.set_cache(cache_type=CacheType.NONE)

    ode_discipline_1 = ODEDiscipline(
        rhs_discipline=discipline,
        times=times,
        state_names={"a": "a_dot", "b": "b_dot"},
        ode_solver_name="DOP853",
        return_trajectories=False,
    )

    res_ode_1 = ode_discipline_1.execute()
    assert_allclose(res_ode_1["final_a"], 1.0)
    assert_allclose(res_ode_1["final_b"], -1.0)

    ode_discipline_2 = ODEDiscipline(
        rhs_discipline=discipline,
        times=times,
        state_names={"a": "a_dot", "b": "b_dot"},
        ode_solver_name="DOP853",
        return_trajectories=False,
    )

    res_ode_2 = ode_discipline_2.execute()
    assert_allclose(res_ode_2["final_a"], 1.0)
    assert_allclose(res_ode_2["final_b"], -1.0)


def test_ode_discipline_missing_names_time_derivatives():
    """Test the error message when the time derivatives are explicitly named,
    but do not correspond to the outputs of the discipline describing the RHS."""
    times = array([0.0, 1.0])

    init_time = array([0.0])
    init_state_a = array([0.0])
    init_state_b = array([0.0])

    def _fct(time=init_time, a=init_state_a, b=init_state_b):
        a_dot = 1.0
        b_dot = -1.0
        return b_dot, a_dot  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=_fct)
    discipline.set_cache(cache_type=CacheType.NONE)

    msg = "'c_dot' are not output variables of the RHS discipline."
    with pytest.raises(ValueError, match=re.escape(msg)):
        ODEDiscipline(
            rhs_discipline=discipline,
            times=times,
            state_names={"a": "c_dot", "b": "b_dot"},
            ode_solver_name="DOP853",
            return_trajectories=False,
        )


def test_ode_discipline_not_convergent():
    """Test the error message when an ODE does not converge to a solution."""
    times = linspace(0.0, 1.0, 20)

    init_time = array([0.0])
    init_state = array([1.0])

    def _fct(time=init_time, x=init_state):
        x_dot = x**2
        return x_dot  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=_fct)
    discipline.set_cache(cache_type=CacheType.NONE)

    ode_discipline = ODEDiscipline(
        rhs_discipline=discipline,
        times=times,
        state_names=["x"],
        ode_solver_name="RK45",
        return_trajectories=True,
    )

    with pytest.raises(
        RuntimeError, match=re.escape("ODE solver RK45 failed to converge.")
    ):
        ode_discipline.execute({"x": array([1.0])})


def test_incompatible_times():
    """Test the error message when ODEDiscipline is provided with a vector of times
    that is not compatible with the problem data."""
    times = linspace(0.0, 10, 30)
    other_times = linspace(0.0, 5, 6)
    ode_disc = OscillatorDiscipline(times=times, omega=4)
    ode_disc.execute()
    ode_disc._ode_problem.result.times = other_times
    with pytest.raises(ValueError) as error_info:
        ode_disc._ode_problem.check()

    msg = "Inconsistent state and time shapes."
    assert msg in str(error_info.value)


# @pytest.mark.parametrize("omega", [0.1, 1.0])
# @pytest.mark.parametrize("init_state_x", [1.0, 0.0])
# @pytest.mark.parametrize("init_state_y", [1.0, 0.0])
def test_jacobian():
    """Test the computation of the Jacobian with respect to the state."""
    omega = 1.0
    init_state_x = 1.0
    init_state_y = 0.0
    init_time = 0.0
    final_time_ = 1.0

    times = linspace(init_time, final_time_, 30)
    init_state_x_ = array([init_state_x])
    init_state_y_ = array([init_state_y])

    def fct(time=init_time, x=init_state_x_, y=init_state_y_):
        x_dot = y
        y_dot = -(omega**2) * x
        return x_dot, y_dot  # noqa: RET504

    def jac_time_state(time=init_time, x=init_state_x_, y=init_state_y_):
        jacobian = array([[0.0, 0.0, 1.0], [0.0, -(omega**2), 0.0]])
        return jacobian  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=fct, py_jac=jac_time_state)
    discipline.set_cache(cache_type=CacheType.NONE)

    ode_discipline = ODEDiscipline(
        rhs_discipline=discipline,
        state_names=["x", "y"],
        ode_solver_name="Radau",
        times=times,
        rtol=1e-12,
        atol=1e-12,
    )

    check_jacobian1 = discipline.check_jacobian(
        input_data={
            "time": array([init_time]),
            "x": init_state_x_,
            "y": init_state_y_,
        },
        input_names=["time", "x", "y"],
        output_names=["x_dot", "y_dot"],
    )
    if not check_jacobian1:
        raise ValueError

    def exact_sol(time):
        if isclose(init_state_x, 0.0, atol=1e-12) and isclose(
            init_state_y, 0.0, atol=1e-12
        ):
            x = 0.0

            y = 0.0
        elif isclose(init_state_x, 0.0, atol=1e-12):
            factor = init_state_y / omega
            x = factor * sin(omega * time)
            y = factor * omega * cos(omega * time)
        else:
            phase = atan(-init_state_y / (omega * init_state_x))
            factor = init_state_x / cos(phase)
            x = factor * cos(omega * time + phase)
            y = -factor * omega * sin(omega * time + phase)
        return x, y

    res_ode = ode_discipline.execute()
    final_time = times[-1]
    x_exact, y_exact = exact_sol(time=final_time)

    assert_allclose(res_ode["final_x"], x_exact)
    assert_allclose(res_ode["final_y"], y_exact)


def test_jacobian_parameters():
    """Test the computation of the Jacobian with respect to the design parameters."""
    omega = 1.0
    init_state_x = 1.0
    init_state_y = 0.0
    init_time = 0.0
    final_time_ = 1.0

    times = linspace(init_time, final_time_, 30)
    init_state_x_ = array([init_state_x])
    init_state_y_ = array([init_state_y])

    def fct(time=init_time, x=init_state_x_, y=init_state_y_, omega=omega):
        x_dot = y
        y_dot = -(omega**2) * x
        return x_dot, y_dot  # noqa: RET504

    def jac_time_state(time=init_time, x=init_state_x_, y=init_state_y_, omega=omega):
        jacobian = array([[0.0, 0.0, 1.0], [0.0, -(omega**2), 0.0]])
        return jacobian  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=fct)
    discipline.set_cache(cache_type=CacheType.NONE)

    ode_discipline = ODEDiscipline(
        rhs_discipline=discipline,
        state_names=["x", "y"],
        ode_solver_name="Radau",
        times=times,
        rtol=1e-12,
        atol=1e-12,
    )

    new_omega = 4.0

    def exact_sol(time):
        if isclose(init_state_x, 0.0, atol=1e-12) and isclose(
            init_state_y, 0.0, atol=1e-12
        ):
            x = 0.0
            y = 0.0
        elif isclose(init_state_x, 0.0, atol=1e-12):
            factor = init_state_y / new_omega
            x = factor * sin(new_omega * time)
            y = factor * new_omega * cos(new_omega * time)
        else:
            phase = atan(-init_state_y / (new_omega * init_state_x))
            factor = init_state_x / cos(phase)
            x = factor * cos(new_omega * time + phase)
            y = -factor * new_omega * sin(new_omega * time + phase)
        return x, y

    res_ode = ode_discipline.execute({"omega": array([new_omega])})
    final_time = times[-1]
    x_exact, y_exact = exact_sol(time=final_time)

    assert_allclose(res_ode["final_x"], x_exact)
    assert_allclose(res_ode["final_y"], y_exact)


@pytest.mark.parametrize(
    "name_of_algorithm",
    ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA", "non_existing_algorithm"],
)
def test_all_ode_integration_algorithms(name_of_algorithm):
    """Test all integration algorithms available in scipy."""
    # times_eval = linspace(0.0, 10.0, 30)
    # omega = 1.0
    times = linspace(0.0, 0.001, 30)
    omega = 100.0
    init_state_x = 1.0
    init_state_y = 0.0

    init_time = array([times[0]])
    init_state_x_ = array([init_state_x])
    init_state_y_ = array([init_state_y])

    def fct(time=init_time, x=init_state_x_, y=init_state_y_):
        x_dot = y
        y_dot = -(omega**2) * x
        return x_dot, y_dot  # noqa: RET504

    def jac(time=init_time, x=init_state_x_, y=init_state_y_):
        jacobian = array([[0.0, 0.0, 1.0], [0.0, -(omega**2), 0.0]])
        return jacobian  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=fct, py_jac=jac)
    discipline.set_cache(cache_type=CacheType.NONE)

    def exact_sol(time):
        if isclose(init_state_x, 0.0, atol=1e-12) and isclose(
            init_state_y, 0.0, atol=1e-12
        ):
            x = 0.0
            y = 0.0
        elif isclose(init_state_x, 0.0, atol=1e-12):
            factor = init_state_y / omega
            x = factor * sin(omega * time)
            y = factor * omega * cos(omega * time)
        else:
            phase = atan(-init_state_y / (omega * init_state_x))
            factor = init_state_x / cos(phase)
            x = factor * cos(omega * time + phase)
            y = -factor * omega * sin(omega * time + phase)
        return x, y

    time_final = times[-1]
    x_exact, y_exact = exact_sol(time=time_final)

    if name_of_algorithm == "non_existing_algorithm":
        with pytest.raises(ValueError) as error_info:
            ODEDiscipline(
                discipline,
                state_names=["x", "y"],
                ode_solver_name=name_of_algorithm,
                times=times,
                rtol=1e-13,
                atol=1e-12,
            )
        assert "No algorithm named non_existing_algorithm is available;" in str(
            error_info.value
        )
    else:
        ode_discipline = ODEDiscipline(
            discipline,
            state_names=["x", "y"],
            ode_solver_name=name_of_algorithm,
            times=times,
            rtol=1e-13,
            atol=1e-12,
        )
        res_ode = ode_discipline.execute()
        assert_allclose(res_ode["final_x"], x_exact)
        assert_allclose(res_ode["final_y"], y_exact)


def test_ode_discipline_termination_event():
    """Test one termination event."""
    times = linspace(0.0, 20.0, 41)
    initial_position = array([10.0])
    initial_velocity = array([0.0])
    initial_time = array([0.0])

    gravity_acceleration = array([-9.81])

    def _rhs_function(
        time=initial_time,
        position=initial_position,
        velocity=initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = gravity_acceleration
        return position_dot, velocity_dot

    def _termination_events_function(
        time=initial_time,
        position=initial_position,
        velocity=initial_velocity,
    ):
        return position

    def _exact_solution(times):
        return initial_position + times * times * gravity_acceleration / 2

    free_fall_discipline = AutoPyDiscipline(py_func=_rhs_function)
    free_fall_discipline.set_cache(cache_type=CacheType.NONE)

    termination_discipline = AutoPyDiscipline(py_func=_termination_events_function)
    termination_discipline.set_cache(cache_type=CacheType.NONE)

    ode_discipline = ODEDiscipline(
        free_fall_discipline,
        times=times,
        state_names=["position", "velocity"],
        ode_solver_name="DOP853",
        return_trajectories=True,
        termination_event_disciplines=(termination_discipline,),
    )

    res_ode = ode_discipline.execute()
    time_evaluation = res_ode["times"]
    exact_solution = _exact_solution(time_evaluation)
    assert allclose(res_ode["position"], exact_solution, rtol=1e-4)
    assert isclose(res_ode["final_position"], 0.0, atol=1e-3)


def test_ode_discipline_two_termination_events():
    """Test two termination events."""
    times = linspace(0.0, 20.0, 41)
    initial_position = array([10.0])
    initial_velocity = array([0.0])
    initial_time = array([0.0])

    gravity_acceleration = array([-9.81])

    def _rhs_function(
        time=initial_time,
        position=initial_position,
        velocity=initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = gravity_acceleration
        return position_dot, velocity_dot

    def _termination_events_function_1(
        time=initial_time,
        position=initial_position,
        velocity=initial_velocity,
    ):
        termination_1 = position
        return termination_1  # noqa: RET504

    def _termination_events_function_2(
        time=initial_time,
        position=initial_position,
        velocity=initial_velocity,
    ):
        termination_2 = 2 * initial_position - position
        return termination_2  # noqa: RET504

    def _exact_solution(times):
        return initial_position + times * times * gravity_acceleration / 2

    free_fall_discipline = AutoPyDiscipline(py_func=_rhs_function)
    free_fall_discipline.set_cache(cache_type=CacheType.NONE)

    termination_discipline_1 = AutoPyDiscipline(py_func=_termination_events_function_1)
    termination_discipline_1.set_cache(cache_type=CacheType.NONE)

    termination_discipline_2 = AutoPyDiscipline(py_func=_termination_events_function_2)
    termination_discipline_2.set_cache(cache_type=CacheType.NONE)

    ode_discipline = ODEDiscipline(
        free_fall_discipline,
        times=times,
        state_names=["position", "velocity"],
        ode_solver_name="DOP853",
        return_trajectories=True,
        termination_event_disciplines=(
            termination_discipline_1,
            termination_discipline_2,
        ),
    )

    res_ode = ode_discipline.execute()
    time_evaluation = res_ode["times"]
    exact_solution = _exact_solution(time_evaluation)
    assert allclose(res_ode["position"], exact_solution, rtol=1e-4)
    assert isclose(res_ode["final_position"], 0.0, atol=1e-3)


def test_serialization(tmp_wd):
    omega = 2
    times = linspace(0.0, 10.0, 41)
    discipline_1 = OscillatorDiscipline(omega, times, return_trajectories=True)
    res_ode_1 = discipline_1.execute()
    to_pickle(discipline_1, "discipline.pkl")
    discipline_2 = from_pickle("discipline.pkl")
    res_ode_2 = discipline_2.execute()
    assert allclose(res_ode_1["position"], res_ode_2["position"])


def test_jacobian_parameters_simple():
    """Test a gradient-based quadrature algorithm."""
    a = 1.0
    x_0 = array([1.0])
    t_0 = 0.0
    t_f = 2.0

    def fct(t=t_0, x=x_0, a=a):
        x_dot = array([a * x])
        return x_dot  # noqa: RET504

    def jacobian(t=t_0, x=x_0, a=a):
        jac = concatenate((t * 0, a, x)).reshape((1, -1))
        return jac  # noqa: RET504

    rhs_discipline = AutoPyDiscipline(py_func=fct, py_jac=jacobian)

    ode_discipline = ODEDiscipline(
        rhs_discipline=rhs_discipline,
        state_names=["x"],
        time_name="t",
        ode_solver_name="Radau",
        times=array([0.0, 1.0]),
    )

    def _exact_sol(tt, x_0):
        return x_0 * exp(a * tt)

    res_ode = ode_discipline.execute({
        "initial_x": x_0,
        "initial_t": t_0,
        "final_t": t_f,
    })
    exact_solution = _exact_sol(t_f, x_0)
    assert isclose(res_ode["final_x"], exact_solution, rtol=1e-4)


def test_swap_order_inputs():
    """Test the case when the inputs of the rhs_discipline are provided in an order
    different from the one in ODEDiscipline."""
    time = array([0.0])
    initial_position_1 = array([1.0])
    initial_velocity_1 = array([0.0])
    omega_1 = array([2.0])

    def rhs_function(
        time=time,
        position=initial_position_1,
        velocity=initial_velocity_1,
        omega=omega_1,
    ):
        position_dot = velocity
        velocity_dot = -(omega**2) * position

        return position_dot, velocity_dot  # noqa: RET504

    rhs_discipline = AutoPyDiscipline(py_func=rhs_function)

    ode_discipline = ODEDiscipline(
        rhs_discipline=rhs_discipline,
        times=linspace(0.0, 1.0, 11),
        state_names=["position", "velocity"],
        return_trajectories=True,
    )
    ode_discipline.execute()
    expected_final_velocity = ode_discipline.io.data["final_velocity"]

    ode_discipline_swap = ODEDiscipline(
        rhs_discipline=rhs_discipline,
        times=linspace(0.0, 1.0, 11),
        # We swap velocity and position.
        state_names=["velocity", "position"],
        return_trajectories=True,
    )
    ode_discipline_swap.execute()
    assert ode_discipline_swap.io.data["final_velocity"] == expected_final_velocity


def test_swap_order_inputs_and_derivatives_with_non_conventional_names():
    """Test the case when the outputs of the rhs_discipline have names
    different from state_dot."""
    time = array([0.0])
    initial_position_1 = array([1.0])
    initial_velocity_1 = array([0.0])
    omega_1 = array([2.0])

    def rhs_function(
        time=time,
        position=initial_position_1,
        velocity=initial_velocity_1,
        omega=omega_1,
    ):
        position_prime = velocity
        velocity_prime = -(omega**2) * position

        return position_prime, velocity_prime  # noqa: RET504

    rhs_discipline = AutoPyDiscipline(py_func=rhs_function)

    ode_discipline_map = ODEDiscipline(
        rhs_discipline=rhs_discipline,
        times=linspace(0.0, 1.0, 51),
        state_names={"velocity": "velocity_prime", "position": "position_prime"},
        return_trajectories=True,
    )
    ode_discipline_map.execute()
    expected_final_position = (initial_velocity_1 / omega_1) * sin(
        omega_1
    ) + initial_position_1 * cos(omega_1)

    tts = linspace(0.0, 1.0, 51)
    (
        (initial_velocity_1 / omega_1) * sin(omega_1 * tts)
        + initial_position_1 * cos(omega_1 * tts)
    )

    assert isclose(
        ode_discipline_map.io.data["final_position"],
        expected_final_position,
        atol=1.0e-3,
    )
