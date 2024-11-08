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
from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import arctan
from numpy import array
from numpy import concatenate
from numpy import cos
from numpy import isclose
from numpy import linspace
from numpy import sin
from numpy import sqrt
from numpy.testing import assert_allclose

from gemseo import create_discipline
from gemseo import from_pickle
from gemseo import to_pickle
from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.problems.ode.oscillator_discipline import OscillatorDiscipline

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline import Discipline


def test_create_oscillator_ode_discipline() -> None:
    """Test the creation of an ODE Discipline."""
    times = linspace(0.0, 10, 30)
    ode_disc = OscillatorDiscipline(times=times, omega=4)
    assert ode_disc is not None


def test_oscillator_ode_discipline_final_time() -> None:
    """Test an ODE Discipline representing a simple oscillator.

    The only part of the solution taken into account is the solution at the final time
    """

    times = linspace(0.0, 10, 30)
    oscillator_discipline = OscillatorDiscipline(omega=2, times=times)
    assert oscillator_discipline is not None

    final_time = times[-1]
    out = oscillator_discipline.execute()
    final_analytical_position = sin(2 * final_time) / 2
    assert allclose(out["position_final"], final_analytical_position)
    final_analytical_velocity = cos(2 * final_time)
    assert allclose(out["velocity_final"], final_analytical_velocity)


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
    assert allclose(out["position_trajectory"], analytical_position)
    analytical_velocity = cos(2 * times)
    assert allclose(out["velocity_trajectory"], analytical_velocity)


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
        "position": initial_position,
        "velocity": initial_velocity,
    })
    analytical_position = sin(omega * times + phase) * amplitude
    analytical_velocity = cos(omega * times + phase) * omega * amplitude
    assert allclose(out["position_trajectory"], analytical_position)
    assert allclose(out["velocity_trajectory"], analytical_velocity)


def test_ode_discipline_bad_grammar() -> None:
    """Test error messages when passing an ill-formed grammar."""
    _initial_time = array([0.0])
    _initial_position = array([0.0])
    _initial_velocity = array([1.0])

    def _rhs_function(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = -4 * position
        return position_dot, velocity_dot

    oscillator = create_discipline(
        "AutoPyDiscipline",
        py_func=_rhs_function,
    )
    with pytest.raises(ValueError) as error_info:
        bad_input_ode_discipline = ODEDiscipline(  # noqa: F841
            discipline=oscillator,
            state_names=["not_position", "not_velocity"],
            times=linspace(0.0, 10, 30),
        )

    assert "Missing default input" in str(error_info.value)


def test_ode_discipline_default_state_names() -> None:
    """Test the assignment of default names to the state variables in ODEDiscipline."""
    _initial_time = array([0.0])
    _initial_position = array([0.0])
    _initial_velocity = array([1.0])

    def _rhs_function(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = -4 * position
        return position_dot, velocity_dot

    oscillator = create_discipline(
        "AutoPyDiscipline",
        py_func=_rhs_function,
    )

    ode_discipline = ODEDiscipline(  # noqa: F841
        discipline=oscillator,
        times=linspace(0.0, 10, 30),
    )

    input_keys = ode_discipline.input_grammar.names
    assert (key in input_keys for key in ("time", "position", "velocity"))


def test_ode_discipline_wrong_ordering_time_derivatives():
    """Test the explicit ordering of the state derivatives in time."""
    _times = array([0.0, 1.0])

    _init_time = array([0.0])
    _init_state_a = array([0.0])
    _init_state_b = array([0.0])

    def _fct(time=_init_time, a=_init_state_a, b=_init_state_b):
        a_dot = 1.0
        b_dot = -1.0
        return b_dot, a_dot  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=_fct)

    ode_discipline_1 = ODEDiscipline(
        discipline=discipline,
        times=_times,
        state_names={"a": "a_dot", "b": "b_dot"},
        ode_solver_name="RK45",
        return_trajectories=False,
    )

    res_ode_1 = ode_discipline_1.execute()
    assert_allclose(res_ode_1["a_final"], 1.0)
    assert_allclose(res_ode_1["b_final"], -1.0)

    ode_discipline_2 = ODEDiscipline(
        discipline=discipline,
        times=_times,
        state_names={"a": "a_dot", "b": "b_dot"},
        ode_solver_name="RK45",
        return_trajectories=False,
    )

    res_ode_2 = ode_discipline_2.execute()
    assert_allclose(res_ode_2["a_final"], 1.0)
    assert_allclose(res_ode_2["b_final"], -1.0)


def test_ode_discipline_missing_names_time_derivatives():
    """Test the error message when the time derivatives are explicitly named, but
    do not correspond to the outputs of the discipline describing the RHS."""
    _times = array([0.0, 1.0])

    _init_time = array([0.0])
    _init_state_a = array([0.0])
    _init_state_b = array([0.0])

    def _fct(time=_init_time, a=_init_state_a, b=_init_state_b):
        a_dot = 1.0
        b_dot = -1.0
        return b_dot, a_dot  # noqa: RET504

    discipline = AutoPyDiscipline(py_func=_fct)

    with pytest.raises(ValueError, match=re.escape("are not names of outputs")):
        ODEDiscipline(
            discipline=discipline,
            times=_times,
            state_names={"a": "c_dot", "b": "b_dot"},
            ode_solver_name="RK45",
            return_trajectories=False,
        )


def test_ode_discipline_not_convergent():
    _times = linspace(0.0, 1.0, 20)

    _init_time = array([0.0])
    _init_state = array([1.0])

    def _fct(time=_init_time, x=_init_state):
        x_dot = x**2
        return x_dot  # noqa: RET504

    discipline = AutoPyDiscipline(
        py_func=_fct,
    )

    ode_discipline = ODEDiscipline(
        discipline=discipline,
        times=_times,
        state_names=["x"],
        ode_solver_name="RK45",
        return_trajectories=True,
    )

    with pytest.raises(
        RuntimeError, match=re.escape("ODE solver RK45 failed to converge.")
    ):
        ode_discipline.execute({"x": array([1.0])})


def test_incompatible_times():
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
    omega = 1.0
    init_state_x = 1.0
    init_state_y = 0.0
    _init_time = 0.0
    _final_time = 1.0
    # _final_time = 10.0

    times = linspace(_init_time, _final_time, 30)
    _init_state_x = array([init_state_x])
    _init_state_y = array([init_state_y])

    def fct(time=_init_time, x=_init_state_x, y=_init_state_y):
        x_dot = y
        y_dot = -(omega**2) * x
        return x_dot, y_dot  # noqa: RET504

    def jac_time_state(time=_init_time, x=_init_state_x, y=_init_state_y):
        jacobian = array([[0.0, 0.0, 1.0], [0.0, -(omega**2), 0.0]])
        return jacobian  # noqa: RET504

    discipline = AutoPyDiscipline(
        py_func=fct,
        py_jac=jac_time_state,
    )

    ode_discipline = ODEDiscipline(
        discipline=discipline,
        state_names=["x", "y"],
        ode_solver_name="Radau",
        times=times,
        rtol=1e-12,
        atol=1e-12,
    )

    check_jacobian1 = discipline.check_jacobian(
        input_data={
            "time": array([_init_time]),
            "x": _init_state_x,
            "y": _init_state_y,
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
    time_final = times[-1]
    x_exact, y_exact = exact_sol(time=time_final)

    assert_allclose(res_ode["x_final"], x_exact)
    assert_allclose(res_ode["y_final"], y_exact)


@pytest.mark.parametrize(
    "name_of_algorithm",
    ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA", "non_existing_algorithm"],
)
def test_all_ode_integration_algorithms(name_of_algorithm):
    # times = linspace(0.0, 10.0, 30)
    # omega = 1.0
    times = linspace(0.0, 0.001, 30)
    omega = 100.0
    init_state_x = 1.0
    init_state_y = 0.0

    _init_time = array([times[0]])
    _init_state_x = array([init_state_x])
    _init_state_y = array([init_state_y])

    def fct(time=_init_time, x=_init_state_x, y=_init_state_y):
        x_dot = y
        y_dot = -(omega**2) * x
        return x_dot, y_dot  # noqa: RET504

    def jac(time=_init_time, x=_init_state_x, y=_init_state_y):
        jacobian = array([[0.0, 0.0, 1.0], [0.0, -(omega**2), 0.0]])
        return jacobian  # noqa: RET504

    discipline = create_discipline(
        "AutoPyDiscipline",
        py_func=fct,
        py_jac=jac,
    )

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
        assert_allclose(res_ode["x_final"], x_exact)
        assert_allclose(res_ode["y_final"], y_exact)


def test_ode_discipline_termination_event():
    _times = linspace(0.0, 20.0, 41)
    _initial_position = array([10.0])
    _initial_velocity = array([0.0])
    _initial_time = array([0.0])

    gravity_acceleration = array([-9.81])

    def _rhs_function(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = gravity_acceleration
        return position_dot, velocity_dot

    def _termination_events_function(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        return position

    def _exact_solution(times):
        return _initial_position + times * times * gravity_acceleration / 2

    free_fall_discipline = create_discipline(
        "AutoPyDiscipline",
        py_func=_rhs_function,
    )

    termination_discipline = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function,
    )

    ode_discipline = ODEDiscipline(
        free_fall_discipline,
        times=_times,
        state_names=["position", "velocity"],
        ode_solver_name="RK45",
        return_trajectories=True,
        termination_event_disciplines=(termination_discipline,),
    )

    res_ode = ode_discipline.execute()
    time_evaluation = res_ode["times"]
    exact_solution = _exact_solution(time_evaluation)
    assert allclose(res_ode["position_trajectory"], exact_solution, rtol=1e-4)
    assert isclose(res_ode["position_final"], 0.0, atol=1e-3)


def test_ode_discipline_two_termination_events_1():
    _times = linspace(0.0, 20.0, 41)
    _initial_position = array([10.0])
    _initial_velocity = array([0.0])
    _initial_time = array([0.0])

    gravity_acceleration = array([-9.81])

    def _rhs_function(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = gravity_acceleration
        return position_dot, velocity_dot

    def _termination_events_function_1(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_1 = position
        return termination_1  # noqa: RET504

    def _termination_events_function_2(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_2 = 2 * _initial_position - position
        return termination_2  # noqa: RET504

    def _exact_solution(times):
        return _initial_position + times * times * gravity_acceleration / 2

    free_fall_discipline = create_discipline(
        "AutoPyDiscipline",
        py_func=_rhs_function,
    )

    termination_discipline_1 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_1,
    )

    termination_discipline_2 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_2,
    )

    ode_mdo_func = DisciplineAdapterGenerator(
        discipline=free_fall_discipline
    ).get_function(
        input_names=["time", "position", "velocity"],
        output_names=["position_dot", "velocity_dot"],
    )

    ode_mdo_termination_1 = DisciplineAdapterGenerator(
        discipline=termination_discipline_1
    ).get_function(
        input_names=["time", "position", "velocity"], output_names=["termination_1"]
    )

    ode_mdo_termination_2 = DisciplineAdapterGenerator(
        discipline=termination_discipline_2
    ).get_function(
        input_names=["time", "position", "velocity"], output_names=["termination_2"]
    )

    def func_time_state(time, state):
        return ode_mdo_func.evaluate(concatenate((array([time]), state)))

    def termination_1_time_state(time, state):
        return ode_mdo_termination_1.evaluate(concatenate((array([time]), state)))

    def termination_2_time_state(time, state):
        return ode_mdo_termination_2.evaluate(concatenate((array([time]), state)))

    ode_problem = ODEProblem(
        func=func_time_state,
        initial_state=concatenate((_initial_position, _initial_velocity)),
        times=_times,
        event_functions=[termination_1_time_state, termination_2_time_state],
    )

    res_ode = ODESolverLibraryFactory().execute(
        ode_problem,
        algo_name="RK45",
        rtol=1e-12,
        atol=1e-12,
    )

    time_evaluation = res_ode.times
    exact_solution = _exact_solution(time_evaluation)
    assert allclose(res_ode.state_trajectories[0], exact_solution, rtol=1e-4)
    assert isclose(res_ode.state_trajectories[0][-1], 0.0, atol=1e-3)


def test_ode_discipline_two_termination_events_2():
    _times = linspace(0.0, 20.0, 41)
    _initial_position = array([10.0])
    _initial_velocity = array([0.0])
    _initial_time = array([0.0])

    gravity_acceleration = array([-9.81])

    def _rhs_function(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = gravity_acceleration
        return position_dot, velocity_dot

    def _termination_events_function_1(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_1 = position
        return termination_1  # noqa: RET504

    def _termination_events_function_2(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_2 = 2 * _initial_position - position
        return termination_2  # noqa: RET504

    def _exact_solution(times):
        return _initial_position + times * times * gravity_acceleration / 2

    free_fall_discipline = create_discipline(
        "AutoPyDiscipline",
        py_func=_rhs_function,
    )

    termination_discipline_1 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_1,
    )

    termination_discipline_2 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_2,
    )

    def make_func_from_adapter(adapter):
        def func(time, state):
            return adapter.evaluate(concatenate((array([time]), state)))

        return func

    ode_mdo_func = DisciplineAdapterGenerator(
        discipline=free_fall_discipline
    ).get_function(
        input_names=["time", "position", "velocity"],
        output_names=["position_dot", "velocity_dot"],
    )

    ode_mdo_termination_1 = DisciplineAdapterGenerator(
        discipline=termination_discipline_1
    ).get_function(
        input_names=["time", "position", "velocity"], output_names=["termination_1"]
    )

    ode_mdo_termination_2 = DisciplineAdapterGenerator(
        discipline=termination_discipline_2
    ).get_function(
        input_names=["time", "position", "velocity"], output_names=["termination_2"]
    )

    ode_problem = ODEProblem(
        func=make_func_from_adapter(ode_mdo_func),
        initial_state=concatenate((_initial_position, _initial_velocity)),
        times=_times,
        event_functions=[
            make_func_from_adapter(ode_mdo_termination_1),
            make_func_from_adapter(ode_mdo_termination_2),
        ],
    )

    res_ode = ODESolverLibraryFactory().execute(
        ode_problem,
        algo_name="RK45",
        rtol=1e-12,
        atol=1e-12,
    )

    time_evaluation = res_ode.times
    exact_solution = _exact_solution(time_evaluation)
    assert allclose(res_ode.state_trajectories[0], exact_solution, rtol=1e-4)
    assert isclose(res_ode.state_trajectories[0][-1], 0.0, atol=1e-3)


def test_ode_discipline_two_termination_events_3():
    _times = linspace(0.0, 20.0, 41)
    _initial_position = array([10.0])
    _initial_velocity = array([0.0])
    _initial_time = array([0.0])

    gravity_acceleration = array([-9.81])

    def _rhs_function(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = gravity_acceleration
        return position_dot, velocity_dot

    def _termination_events_function_1(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_1 = position
        return termination_1  # noqa: RET504

    def _termination_events_function_2(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_2 = 2 * _initial_position - position
        return termination_2  # noqa: RET504

    def _exact_solution(times):
        return _initial_position + times * times * gravity_acceleration / 2

    free_fall_discipline = create_discipline(
        "AutoPyDiscipline",
        py_func=_rhs_function,
    )

    termination_discipline_1 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_1,
    )

    termination_discipline_2 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_2,
    )

    def make_func_from_adapter(adapter):
        def func(time, state):
            return adapter.evaluate(concatenate((array([time]), state)))

        return func

    ode_mdo_func = DisciplineAdapterGenerator(
        discipline=free_fall_discipline
    ).get_function(
        input_names=["time", "position", "velocity"],
        output_names=["position_dot", "velocity_dot"],
    )

    ode_mdo_termination_1 = DisciplineAdapterGenerator(
        discipline=termination_discipline_1
    ).get_function(
        input_names=["time", "position", "velocity"], output_names=["termination_1"]
    )

    ode_mdo_termination_2 = DisciplineAdapterGenerator(
        discipline=termination_discipline_2
    ).get_function(
        input_names=["time", "position", "velocity"], output_names=["termination_2"]
    )

    adapter_events = [ode_mdo_termination_1, ode_mdo_termination_2]

    ode_problem = ODEProblem(
        func=make_func_from_adapter(ode_mdo_func),
        initial_state=concatenate((_initial_position, _initial_velocity)),
        times=_times,
        event_functions=[make_func_from_adapter(event) for event in adapter_events],
    )

    res_ode = ODESolverLibraryFactory().execute(
        ode_problem,
        algo_name="RK45",
        rtol=1e-12,
        atol=1e-12,
    )

    time_evaluation = res_ode.times
    exact_solution = _exact_solution(time_evaluation)
    assert allclose(res_ode.state_trajectories[0], exact_solution, rtol=1e-4)
    assert isclose(res_ode.state_trajectories[0][-1], 0.0, atol=1e-3)


def test_ode_discipline_two_termination_events_4():
    _times = linspace(0.0, 20.0, 41)
    _initial_position = array([10.0])
    _initial_velocity = array([0.0])
    _initial_time = array([0.0])

    gravity_acceleration = array([-9.81])

    def _rhs_function(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = gravity_acceleration
        return position_dot, velocity_dot

    def _termination_events_function_1(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_1 = position
        return termination_1  # noqa: RET504

    def _termination_events_function_2(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_2 = 2 * _initial_position - position
        return termination_2  # noqa: RET504

    def _exact_solution(times):
        return _initial_position + times * times * gravity_acceleration / 2

    free_fall_discipline = create_discipline(
        "AutoPyDiscipline",
        py_func=_rhs_function,
    )

    termination_discipline_1 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_1,
    )

    termination_discipline_2 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_2,
    )

    def make_func_from_discipline(disc: Discipline):
        adapter = DisciplineAdapterGenerator(discipline=disc).get_function(
            input_names=disc.io.input_grammar.names,
            output_names=disc.io.output_grammar.names,
        )

        def func(time, state):
            return adapter.evaluate(concatenate((array([time]), state)))

        return func

    disc_events = [termination_discipline_1, termination_discipline_2]

    ode_problem = ODEProblem(
        func=make_func_from_discipline(free_fall_discipline),
        initial_state=concatenate((_initial_position, _initial_velocity)),
        times=_times,
        event_functions=[make_func_from_discipline(event) for event in disc_events],
    )

    res_ode = ODESolverLibraryFactory().execute(
        ode_problem,
        algo_name="RK45",
        rtol=1e-12,
        atol=1e-12,
    )

    time_evaluation = res_ode.times
    exact_solution = _exact_solution(time_evaluation)
    assert allclose(res_ode.state_trajectories[0], exact_solution, rtol=1e-4)
    assert isclose(res_ode.state_trajectories[0][-1], 0.0, atol=1e-3)


def test_ode_discipline_two_termination_events_6():
    _times = linspace(0.0, 20.0, 41)
    _initial_position = array([10.0])
    _initial_velocity = array([0.0])
    _initial_time = array([0.0])

    gravity_acceleration = array([-9.81])

    def _rhs_function(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        position_dot = velocity
        velocity_dot = gravity_acceleration
        return position_dot, velocity_dot

    def _termination_events_function_1(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_1 = position
        return termination_1  # noqa: RET504

    def _termination_events_function_2(
        time=_initial_time,
        position=_initial_position,
        velocity=_initial_velocity,
    ):
        termination_2 = 2 * _initial_position - position
        return termination_2  # noqa: RET504

    def _exact_solution(times):
        return _initial_position + times * times * gravity_acceleration / 2

    free_fall_discipline = create_discipline(
        "AutoPyDiscipline",
        py_func=_rhs_function,
    )

    termination_discipline_1 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_1,
    )

    termination_discipline_2 = create_discipline(
        "AutoPyDiscipline",
        py_func=_termination_events_function_2,
    )

    ode_discipline = ODEDiscipline(
        free_fall_discipline,
        times=_times,
        state_names=["position", "velocity"],
        ode_solver_name="RK45",
        return_trajectories=True,
        termination_event_disciplines=(
            termination_discipline_1,
            termination_discipline_2,
        ),
    )

    res_ode = ode_discipline.execute()
    time_evaluation = res_ode["times"]
    exact_solution = _exact_solution(time_evaluation)
    assert allclose(res_ode["position_trajectory"], exact_solution, rtol=1e-4)
    assert isclose(res_ode["position_final"], 0.0, atol=1e-3)


def test_serialization(tmp_wd):
    omega = 2
    times = linspace(0.0, 10.0, 41)
    discipline_1 = OscillatorDiscipline(omega, times, return_trajectories=True)
    res_ode_1 = discipline_1.execute()
    to_pickle(discipline_1, "discipline.pkl")
    discipline_2 = from_pickle("discipline.pkl")
    res_ode_2 = discipline_2.execute()
    assert allclose(res_ode_1["position_trajectory"], res_ode_2["position_trajectory"])
