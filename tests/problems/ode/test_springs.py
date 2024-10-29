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
"""Tests of the problem of masses connected by springs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import concatenate
from numpy import linspace
from numpy import ndarray
from numpy import repeat
from numpy import sin

from gemseo import create_discipline
from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.core.chains.chain import MDOChain
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.problems.ode._springs import Mass
from gemseo.problems.ode._springs import compute_analytic_mass_position
from gemseo.problems.ode._springs import create_chained_masses
from gemseo.problems.ode._springs import create_mass_discipline
from gemseo.problems.ode._springs import create_mass_ode_discipline
from gemseo.problems.ode._springs import generic_mass_rhs_function

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.typing import NumberArray


def test_create_mass_discipline():
    times = linspace(0, 10, 30)
    discipline = create_mass_discipline(mass=1, left_stiff=1, right_stiff=1)

    ode_discipline = ODEDiscipline(
        discipline=discipline,
        state_names=["position", "velocity"],
        times=times,
    )
    ode_discipline.execute()


def test_create_mass_ode_discipline():
    times = linspace(0, 10, 30)
    ode_discipline = create_mass_ode_discipline(
        mass=1,
        left_stiff=1,
        right_stiff=1,
        times=times,
        state_names=["position_1", "velocity_1"],
        state_dot_var_names=["position_1_dot", "velocity_1_dot"],
        is_left_pos_fixed=True,
        is_right_pos_fixed=True,
        rtol=1e-12,
        atol=1e-12,
    )
    ode_discipline.execute()


def test_create_mass_ode_discipline_without_options():
    times = linspace(0, 10, 30)
    ode_discipline = create_mass_ode_discipline(
        mass=1,
        left_stiff=1,
        right_stiff=1,
        times=times,
        state_names=["position_1", "velocity_1"],
        state_dot_var_names=["position_1_dot", "velocity_1_dot"],
        is_left_pos_fixed=True,
        is_right_pos_fixed=True,
    )
    ode_discipline.execute()


# @pytest.mark.parametrize(
#     ("left_stiff", "right_stiff", "initial_position", "initial_velocity", "mass"),
#     [
#         (1.0, 1.0, 0.0, 1.0, 1.0),
#         (10.0, 1.0, 0.0, 1.0, 1.0),
#         (1.0, 1.0, 0.0, 1.0, 10.0),
#         (10.0, 1.0, 0.0, 1.0, 10.0),
#         (1.0, 10.0, 0.0, 1.0, 1.0),
#         (1.0, 10.0, 1.0, 1.0, 1.0),
#         (1.0, 1.0, 0.0, 10.0, 1.0),
#         (10.0, 10.0, 1.0, 0.0, 1.0),
#         (1.0, 10.0, 1.0, 1.0, 10.0),
#         (1.0, 1.0, 0.0, 10.0, 10.0),
#         (10.0, 10.0, 1.0, 0.0, 10.0),
#     ],
# )
def test_generic_mass_rhs_function():
    """Test the resolution for a single mass connected by springs to fixed points.

    Verify the values of the output for various initial conditions.
    """
    left_stiff = 1.0
    right_stiff = 1.0
    initial_position = 0.0
    initial_velocity = 1.0
    mass = 10.0
    times = linspace(0, 5, 30)
    left_position = 0.0
    right_position = 0.0

    ode_discipline = create_mass_ode_discipline(
        mass=mass,
        left_stiff=left_stiff,
        right_stiff=right_stiff,
        left_position=left_position,
        right_position=right_position,
        times=times,
        state_names=["position", "velocity"],
        is_left_pos_fixed=True,
        is_right_pos_fixed=True,
        rtol=1e-12,
        atol=1e-12,
    )

    assert ode_discipline is not None

    result = ode_discipline.execute({
        "position": array([initial_position]),
        "velocity": array([initial_velocity]),
    })

    def ode_rhs(time: NumberArray, state: NDArray[NumberArray, NumberArray]) -> ndarray:
        position_dot, velocity_dot = generic_mass_rhs_function(
            time=time,
            position=state[0],
            velocity=state[1],
            mass=mass,
            left_stiff=left_stiff,
            right_stiff=right_stiff,
            left_position=left_position,
            right_position=right_position,
            times=times,
        )
        return array([position_dot, velocity_dot])

    ode_problem = ODEProblem(
        ode_rhs,
        initial_state=array([initial_position, initial_velocity]),
        times=times,
    )
    ODESolverLibraryFactory().execute(
        ode_problem, "RK45", first_step=1e-6, rtol=1e-12, atol=1e-12
    )
    analytical_position_trajectory = compute_analytic_mass_position(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        left_stiff=left_stiff,
        right_stiff=right_stiff,
        mass=mass,
        times=times,
    )

    assert allclose(result["position_trajectory"], analytical_position_trajectory)


# @pytest.mark.parametrize("stiff_1", [1.0, 0.1])
# @pytest.mark.parametrize("stiff_2", [1.0, 0.1])
# @pytest.mark.parametrize("mass_value_1", [1.0, 10])
def test_2_chained_masses():
    """Tests the coupling two disciplines representing two masses connected by a spring
    to one another and to foxed walls. The coupling is performed by MDAGaussSeidel.

    The result is compared with the classical numerical solution of the mass-spring
    problem. Different values for the first mass and th stiffnesses of the first two
    springs are tested.
    """
    stiff_1 = 0.1
    stiff_2 = 0.1
    mass_value_1 = 10.0
    _time_init = array([0.0])
    times = linspace(0.0, 10, 60)
    mass_value_2 = 1.0
    stiff_3 = 1.0

    position_0_init = array([1.0])
    velocity_0_init = array([0.0])
    position_1_init = array([0.0])
    velocity_1_init = array([0.0])
    position_0_sol_init = repeat(position_0_init, len(times))
    position_1_sol_init = repeat(position_1_init, len(times))

    def mass_0_rhs(
        time=_time_init,
        position_0=position_0_init,
        velocity_0=velocity_0_init,
        position_1_trajectory=position_1_sol_init,
    ):
        if isinstance(position_1_trajectory, float):
            right_pos = array([position_1_trajectory] * len(times))
        elif len(position_1_trajectory) == 1:
            right_pos = repeat(position_1_trajectory, len(times))
        else:
            right_pos = position_1_trajectory

        position_0_dot, velocity_0_dot = generic_mass_rhs_function(
            time=time,
            position=position_0,
            velocity=velocity_0,
            mass=mass_value_1,
            left_stiff=stiff_1,
            right_stiff=stiff_2,
            left_position=0,
            right_position=right_pos,
            times=times,
        )
        return position_0_dot, velocity_0_dot

    def mass_1_rhs(
        time=_time_init,
        position_1=position_1_init,
        velocity_1=velocity_1_init,
        position_0_trajectory=position_0_sol_init,
    ):
        if isinstance(position_0_trajectory, float):
            left_pos = array([position_0_trajectory] * len(times))
        elif len(position_0_trajectory) == 1:
            left_pos = repeat(position_0_trajectory, len(times))
        else:
            left_pos = position_0_trajectory

        position_1_dot, velocity_1_dot = generic_mass_rhs_function(
            time=time,
            position=position_1,
            velocity=velocity_1,
            mass=mass_value_2,
            left_stiff=stiff_2,
            right_stiff=stiff_3,
            left_position=left_pos,
            right_position=0,
            times=times,
        )
        return position_1_dot, velocity_1_dot

    discipline_0 = create_discipline(
        "AutoPyDiscipline",
        py_func=mass_0_rhs,
    )
    ode_discipline_0 = ODEDiscipline(
        discipline=discipline_0,
        state_names=["position_0", "velocity_0"],
        times=times,
        return_trajectories=True,
        rtol=1e-6,
        atol=1e-6,
    )

    discipline1 = create_discipline(
        "AutoPyDiscipline",
        py_func=mass_1_rhs,
    )
    ode_discipline1 = ODEDiscipline(
        discipline=discipline1,
        state_names=["position_1", "velocity_1"],
        times=times,
        return_trajectories=True,
        rtol=1e-6,
        atol=1e-6,
    )

    disciplines = [ode_discipline_0, ode_discipline1]
    mda = MDAGaussSeidel(disciplines, tolerance=1e-6)

    discipline_result = mda.execute({
        "position_0": position_0_init,
        "position_1": position_1_init,
        "position_0_trajectory": position_0_init,
        "position_1_trajectory": position_1_init,
    })

    assert sorted(mda.coupling_structure.strong_couplings) == sorted([
        "position_0_trajectory",
        "position_1_trajectory",
    ])

    def _ode_func(time: NumberArray, state: NumberArray) -> NumberArray:
        position_0_dot = state[1]
        velocity_0_dot = (
            -(stiff_1 + stiff_2) / mass_value_1 * state[0]
            + stiff_2 / mass_value_1 * state[2]
        )
        position_1_dot = state[3]
        velocity_1_dot = (
            -(stiff_2 + stiff_3) / mass_value_2 * state[2]
            + stiff_2 / mass_value_2 * state[0]
        )
        return array([position_0_dot, velocity_0_dot, position_1_dot, velocity_1_dot])

    ode_problem = ODEProblem(
        _ode_func,
        initial_state=concatenate([
            position_0_init,
            velocity_0_init,
            position_1_init,
            velocity_1_init,
        ]),
        times=times,
    )
    ODESolverLibraryFactory().execute(ode_problem, "RK45", rtol=1e-6, atol=1e-6)

    assert allclose(
        ode_problem.result.state_trajectories[0],
        discipline_result["position_0_trajectory"],
        atol=5.0e-2,
    )

    assert allclose(
        ode_problem.result.state_trajectories[2],
        discipline_result["position_1_trajectory"],
        atol=5.0e-2,
    )


def test_2_chained_masses_linear_coupling():
    """Test the chained masses problem.

    IDF version of the problem with two masses.
    """
    times = linspace(0.0, 10, 30)
    mass_value_1 = 1
    mass_value_2 = 1
    stiff_1 = 1
    stiff_2 = 1
    stiff_3 = 1

    position_1_sol_init = array([1.0])
    velocity_1_sol_init = array([0.0])
    position_2_sol_init = array([0.0])
    velocity_2_sol_init = array([0.0])

    def mass_1_rhs(
        time=0,
        position_1=position_1_sol_init,
        velocity_1=velocity_1_sol_init,
        position_2_sol=position_2_sol_init,
    ):
        position_1_dot, velocity_1_dot = generic_mass_rhs_function(
            time=array([time]),
            position=array([position_1]),
            velocity=array([velocity_1]),
            mass=mass_value_1,
            left_stiff=stiff_1,
            right_stiff=stiff_2,
            left_position=0,
            right_position=position_2_sol,
            times=times,
        )
        return position_1_dot, velocity_1_dot

    def mass_2_rhs(
        time=0,
        position_2=position_2_sol_init,
        velocity_2=velocity_2_sol_init,
        position_1_sol=position_1_sol_init,
    ):
        position_2_dot, velocity_2_dot = generic_mass_rhs_function(
            time=array([time]),
            position=array([position_2]),
            velocity=array([velocity_2]),
            mass=mass_value_2,
            left_stiff=stiff_2,
            right_stiff=stiff_3,
            left_position=0,
            right_position=position_1_sol,
            times=times,
        )
        return position_2_dot, velocity_2_dot

    discipline_1 = create_discipline(
        "AutoPyDiscipline",
        py_func=mass_1_rhs,
    )
    discipline_2 = create_discipline(
        "AutoPyDiscipline",
        py_func=mass_2_rhs,
    )
    mda = MDOChain(
        [discipline_1, discipline_2],
    )
    ode_discipline = ODEDiscipline(
        discipline=mda,
        state_names=["position_1", "velocity_1", "position_2", "velocity_2"],
        times=times,
        return_trajectories=True,
        rtol=1e-12,
        atol=1e-12,
    )
    discipline_result = ode_discipline.execute()

    def _ode_func(time: NumberArray, state: NumberArray) -> NumberArray:
        position_1_dot = state[1]
        velocity_1_dot = -(stiff_1 + stiff_2) / mass_value_1 * state[0]
        return array([position_1_dot, velocity_1_dot])

    ode_problem = ODEProblem(
        _ode_func,
        times=times,
        initial_state=concatenate([position_1_sol_init, velocity_1_sol_init]),
    )
    ODESolverLibraryFactory().execute(
        ode_problem,
        "RK45",
        rtol=1e-12,
        atol=1e-12,
    )

    assert allclose(
        ode_problem.result.state_trajectories[0],
        discipline_result["position_1_trajectory"],
        atol=1e-6,
    )


def test_create_chained_masses():
    chain = create_chained_masses(
        1.0,
        Mass(mass=1.0, position=1.0, left_stiffness=1.0),
        Mass(mass=2.0, position=0.0, left_stiffness=1.0),
        Mass(mass=3.0, position=0.0, left_stiffness=1.0),
    )
    mda = MDOChain(chain)
    mda.execute()


# @pytest.mark.parametrize("stiff_1", [1.0, 0.1])
# @pytest.mark.parametrize("stiff_2", [1.0, 0.1])
# @pytest.mark.parametrize("mass_value_1", [1.0, 10.0])
def test_create_two_chained_masses():
    """Tests the coupling two disciplines representing two masses connected by a spring
    to one another and to foxed walls. The coupling is performed by MDAGaussSeidel.

    The disciplines representing the chained masses are created by the
    `create_chained_masses` method.

    The result is compared with the classical numerical solution of the mass-spring
    problem. Different values for the first mass and th stiffnesses of the first two
    springs are tested.
    """
    stiff_1 = 0.1
    stiff_2 = 0.1
    mass_value_1 = 10.0
    masses = [mass_value_1, 1.0]
    stiffnesses = [stiff_1, stiff_2, 1.0]
    positions = [1.0, 0.0]
    times = linspace(0.0, 10, 60)
    chain = create_chained_masses(
        1.0,
        Mass(mass=mass_value_1, position=1.0, left_stiffness=stiff_1),
        Mass(mass=1.0, position=0.0, left_stiffness=stiff_2),
        times=times,
        rtol=1e-8,
        atol=1e-8,
    )

    mda = MDAGaussSeidel(chain, tolerance=1e-8)

    discipline_result = mda.execute({
        "position0": array([positions[0]]),
        "position1": array([positions[1]]),
    })

    def _ode_func(time: NumberArray, state: NumberArray) -> NumberArray:
        position_0_dot = state[1]
        velocity_0_dot = (
            -(stiffnesses[0] + stiffnesses[1]) / masses[0] * state[0]
            + stiffnesses[1] / masses[0] * state[2]
        )
        position_1_dot = state[3]
        velocity_1_dot = (
            -(stiffnesses[1] + stiffnesses[2]) / masses[1] * state[2]
            + stiffnesses[1] / masses[1] * state[0]
        )
        return array([position_0_dot, velocity_0_dot, position_1_dot, velocity_1_dot])

    ode_problem = ODEProblem(
        _ode_func,
        initial_state=array([positions[0], 0.0, positions[1], 0.0]),
        times=times,
    )
    ODESolverLibraryFactory().execute(
        ode_problem,
        "RK45",
        rtol=1e-6,
        atol=1e-6,
    )

    assert allclose(
        ode_problem.result.state_trajectories[0],
        discipline_result["position0_trajectory"],
        atol=5e-2,
    )

    assert allclose(
        ode_problem.result.state_trajectories[2],
        discipline_result["position1_trajectory"],
        atol=5e-2,
    )


def test_lateral_masses_as_floats():
    """Test the case when the position of the lateral masses are initialized as
    floats."""

    times = linspace(0.0, 10, 30)
    ode_discipline_1 = create_mass_ode_discipline(
        mass=1,
        left_stiff=1,
        right_stiff=1,
        times=times,
        state_names=["position_1", "velocity_1"],
        state_dot_var_names=["position_1_dot", "velocity_1_dot"],
        left_position=0.0,
        right_position=0.0,
        left_position_name="left_position",
        right_position_name="right_position",
        is_left_pos_fixed=False,
        is_right_pos_fixed=False,
        rtol=1e-12,
        atol=1e-12,
    )
    ode_discipline_1.execute()


def test_one_mass_attached_to_moving_pins():
    """Test the case where a single mass is linked to mobile pins to its left and right,
    moving according to a given trajectory."""
    mass = 1.0
    left_stiff = 1.0
    right_stiff = 1.0
    omega = 1.0
    times = linspace(0, 2, 30)

    def fct_left_position(t: NumberArray | float = 0.0):
        return sin(omega * t)

    def fct_right_position(t: NumberArray | float = 0.0):
        return sin(omega * t)

    left_position = fct_left_position(times)
    right_position = fct_right_position(times)

    ode_discipline = create_mass_ode_discipline(
        mass=mass,
        left_stiff=left_stiff,
        right_stiff=right_stiff,
        left_position=left_position,
        right_position=right_position,
        times=times,
        state_names=["position", "velocity"],
        rtol=1e-12,
        atol=1e-12,
    )

    discipline_result = ode_discipline.execute({
        "position": array([0.0]),
        "velocity": array([0.0]),
    })

    def _ode_func(time: NumberArray, state: NumberArray) -> NumberArray:
        position_dot = state[1]
        velocity_dot = (
            -(left_stiff + right_stiff) / mass * state[0]
            + left_stiff * fct_left_position(time)
            + right_stiff * fct_right_position(time)
        )
        return array([position_dot, velocity_dot])

    ode_problem = ODEProblem(
        _ode_func,
        times=times,
        initial_state=array([0.0, 0.0]),
    )
    ODESolverLibraryFactory().execute(
        ode_problem,
        "RK45",
        rtol=1e-12,
        atol=1e-12,
    )

    assert allclose(
        ode_problem.result.state_trajectories[0],
        discipline_result["position_trajectory"],
        atol=1e-3,
    )


def test_one_mass_attached_to_mobile_pins_wrong_time_lengths():
    """Test the error messages when the positions provided for the left and right masses
    are incompatible with times."""
    mass = 1.0
    left_stiff = 1.0
    right_stiff = 1.0
    omega = 1.0
    times = linspace(0, 2, 30)
    other_times = linspace(0, 2, 20)

    def fct_left_position(t: NumberArray | float = 0.0):
        return sin(omega * t)

    def fct_right_position(t: NumberArray | float = 0.0):
        return sin(omega * t)

    left_position = fct_left_position(other_times)
    right_position = fct_right_position(times)

    with pytest.raises(ValueError) as err:
        create_mass_ode_discipline(
            mass=mass,
            left_stiff=left_stiff,
            right_stiff=right_stiff,
            left_position=left_position,
            right_position=right_position,
            times=times,
            state_names=["position", "velocity"],
            rtol=1e-12,
            atol=1e-12,
        )
    assert "Incoherent lengths" in str(err.value)

    left_position = fct_left_position(times)
    right_position = fct_right_position(other_times)

    with pytest.raises(ValueError) as err:
        create_mass_ode_discipline(
            mass=mass,
            left_stiff=left_stiff,
            right_stiff=right_stiff,
            left_position=left_position,
            right_position=right_position,
            times=times,
            state_names=["position", "velocity"],
            rtol=1e-12,
            atol=1e-12,
        )
    assert "Incoherent lengths" in str(err.value)
