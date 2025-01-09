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
from numpy import repeat
from numpy import sin
from numpy.testing import assert_allclose

from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.ode.ode_problem import ODEProblem
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.problems.ode.springs.coupled_springs_generator import (
    CoupledSpringsGenerator,
)
from gemseo.problems.ode.springs.spring_ode_discipline import SpringODEDiscipline
from gemseo.problems.ode.springs.springs_dynamics_discipline import (
    SpringsDynamicsDiscipline,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


def test_create_mass_mdo_discipline():
    mdo_discipline = SpringsDynamicsDiscipline(
        mass=1,
        left_stiffness=1,
        right_stiffness=1,
        is_left_position_fixed=True,
        is_right_position_fixed=True,
    )
    mdo_discipline.execute()


def test_create_mass_ode_discipline():
    times = linspace(0, 10, 30)
    ode_discipline = SpringODEDiscipline(
        mass=1,
        left_stiffness=1,
        right_stiffness=1,
        times=times,
        state_names=("position_1", "velocity_1"),
        state_dot_names=("position_1_dot", "velocity_1_dot"),
        is_left_position_fixed=True,
        is_right_position_fixed=True,
        rtol=1e-12,
        atol=1e-12,
    )
    ode_discipline.execute()


def test_create_mass_ode_discipline_without_options():
    times = linspace(0, 10, 30)
    ode_discipline = SpringODEDiscipline(
        mass=1,
        left_stiffness=1,
        right_stiffness=1,
        times=times,
        state_names=("position_1", "velocity_1"),
        state_dot_names=("position_1_dot", "velocity_1_dot"),
        is_left_position_fixed=True,
        is_right_position_fixed=True,
    )
    ode_discipline.execute()


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

    ode_discipline = SpringODEDiscipline(
        mass=mass,
        left_stiffness=left_stiff,
        right_stiffness=right_stiff,
        left_position=left_position,
        right_position=right_position,
        times=times,
        state_names=("position", "velocity"),
        is_left_position_fixed=True,
        is_right_position_fixed=True,
        rtol=1e-12,
        atol=1e-12,
    )

    assert ode_discipline is not None

    result = ode_discipline.execute({
        "initial_position": array([initial_position]),
        "initial_velocity": array([initial_velocity]),
    })

    analytical_position_trajectory = SpringODEDiscipline.compute_analytic_mass_position(
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        left_stiffness=left_stiff,
        right_stiffness=right_stiff,
        mass=mass,
        times=times,
    )

    assert allclose(result["position"], analytical_position_trajectory)


# @pytest.mark.parametrize("stiff_1", [1.0, 0.1])
# @pytest.mark.parametrize("stiff_2", [1.0, 0.1])
# @pytest.mark.parametrize("mass_value_1", [1.0, 10])
def test_2_chained_masses():
    """Tests the coupling two disciplines representing two masses connected by a spring
    to one another and to fixed walls. The coupling is performed by MDAGaussSeidel.

    The result is compared with the classical numerical solution of the mass-spring
    problem. Different values for the first mass and th stiffnesses of the first two
    springs are tested.
    """
    stiff_1 = 0.1
    stiff_2 = 0.1
    mass_value_0 = 10.0
    _time_init = array([0.0])
    times = linspace(0.0, 10, 60)
    mass_value_1 = 1.0
    stiff_3 = 1.0

    position_0_init = array([1.0])
    velocity_0_init = array([0.0])
    position_1_init = array([0.0])
    velocity_1_init = array([0.0])
    position_0_trajectory_init = repeat(position_0_init, len(times))
    position_1_trajectory_init = repeat(position_1_init, len(times))

    ode_discipline_0 = SpringODEDiscipline(
        mass=mass_value_0,
        left_stiffness=stiff_1,
        right_stiffness=stiff_2,
        times=times,
        left_position=0.0,
        state_names=("position_0", "velocity_0"),
        right_position_name="position_1",
        is_left_position_fixed=True,
        is_right_position_fixed=False,
    )

    ode_discipline_1 = SpringODEDiscipline(
        mass=mass_value_1,
        left_stiffness=stiff_2,
        right_stiffness=stiff_3,
        times=times,
        right_position=0.0,
        state_names=("position_1", "velocity_1"),
        left_position_name="position_0",
        is_left_position_fixed=False,
        is_right_position_fixed=True,
    )

    disciplines = [ode_discipline_0, ode_discipline_1]
    mda = MDAGaussSeidel(disciplines, tolerance=1e-12)

    discipline_result = mda.execute({
        "initial_position_0": position_0_init,
        "initial_position_1": position_1_init,
        "position_0": position_0_trajectory_init,
        "position_1": position_1_trajectory_init,
    })

    assert sorted(mda.coupling_structure.strong_couplings) == sorted([
        "position_0",
        "position_1",
        "times",
    ])

    def _ode_func(time: RealArray, state: RealArray) -> RealArray:
        position_0_dot = state[1]
        velocity_0_dot = (
            -(stiff_1 + stiff_2) / mass_value_0 * state[0]
            + stiff_2 / mass_value_0 * state[2]
        )
        position_1_dot = state[3]
        velocity_1_dot = (
            -(stiff_2 + stiff_3) / mass_value_1 * state[2]
            + stiff_2 / mass_value_1 * state[0]
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
    ODESolverLibraryFactory().execute(
        ode_problem, algo_name="Radau", rtol=1e-6, atol=1e-6
    )

    assert allclose(
        ode_problem.result.state_trajectories[0],
        discipline_result["position_0"],
        atol=5.0e-2,
    )

    assert allclose(
        ode_problem.result.state_trajectories[2],
        discipline_result["position_1"],
        atol=5.0e-2,
    )


def test_chained_masses():
    """Test the two methods to reproduce the movement of three masses connected by
    springs, by performing an MDA on coupled ODEDisciplines and by defining a single
    ODEDiscipline whose dynamic is the result of an MDA."""
    masses = [1.0, 2.0, 3.0]
    stiffnesses = [1.0, 1.0, 1.0, 1.0]
    initial_positions = [1.0, 0.0, 0.0]
    springs_and_masses = CoupledSpringsGenerator(masses=masses, stiffnesses=stiffnesses)

    mda = MDAGaussSeidel(springs_and_masses.create_coupled_ode_disciplines())
    mda_result = mda.execute({
        initial_position_name: array([initial_positions[i]])
        for i, initial_position_name in enumerate(
            springs_and_masses.initial_position_names
        )
    })

    ode = springs_and_masses.create_discipline_with_coupled_dynamics()
    ode_result = ode.execute({
        initial_position_name: array([initial_positions[i]])
        for i, initial_position_name in enumerate(
            springs_and_masses.initial_position_names
        )
    })

    def _ode_func(time: RealArray, state: RealArray) -> RealArray:
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
        initial_state=array([initial_positions[0], 0.0, initial_positions[1], 0.0]),
        times=springs_and_masses.times,
    )
    ODESolverLibraryFactory().execute(
        ode_problem,
        algo_name="RK45",
        rtol=1e-6,
        atol=1e-6,
    )

    for name in springs_and_masses.position_names:
        name_trajectory = f"{name}"
        assert_allclose(
            mda_result[name_trajectory], ode_result[name_trajectory], atol=5e-2
        )


def test_create_two_chained_masses():
    """Tests the coupling two disciplines representing two masses connected
    by a spring to one another and to fixed walls.
    The coupling is performed by MDAGaussSeidel.

    The disciplines representing the chained masses are created by the
    `create_chained_masses` method.

    The result is compared with the classical numerical solution of the mass-spring
    problem. Different values for the first mass and th stiffnesses of the first two
    springs are tested.
    """

    masses = [2.0, 1.0]
    stiffnesses = [2.0, 2.0, 1.0]
    positions = [1.0, 0.0]
    times = linspace(0.0, 10, 60)

    springs_and_masses = CoupledSpringsGenerator(
        masses=masses, stiffnesses=stiffnesses, times=times
    )

    mda = MDAGaussSeidel(springs_and_masses.create_coupled_ode_disciplines(atol=1e-8))

    discipline_result = mda.execute({
        "initial_position_0": array([positions[0]]),
        "initial_position_1": array([positions[1]]),
    })

    def _ode_func(time: RealArray, state: RealArray) -> RealArray:
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
        algo_name="Radau",
        rtol=1e-6,
        atol=1e-6,
    )

    assert allclose(
        ode_problem.result.state_trajectories[0],
        discipline_result["position_0"],
        atol=5e-2,
    )

    assert allclose(
        ode_problem.result.state_trajectories[2],
        discipline_result["position_1"],
        atol=5e-2,
    )


def test_create_ode_discipline_for_two_masses():
    """Tests the coupling two disciplines representing two masses connected
    by a spring to one another and to foxed walls.
    The coupling is performed by MDAGaussSeidel.

    The disciplines representing the chained masses are created by the
    `create_chained_masses` method.

    The result is compared with the classical numerical solution of the mass-spring
    problem. Different values for the first mass and th stiffnesses of the first two
    springs are tested.
    """
    masses = [10.0, 1.0]
    stiffnesses = [0.1, 0.1, 1.0]
    positions = [1.0, 0.0]
    times = linspace(0.0, 10, 60)

    springs_and_masses = CoupledSpringsGenerator(
        masses=masses, stiffnesses=stiffnesses, times=times
    )

    ode_discipline = springs_and_masses.create_discipline_with_coupled_dynamics()

    discipline_result = ode_discipline.execute({
        "initial_position_0": array([positions[0]]),
        "initial_position_1": array([positions[1]]),
    })

    def _ode_func(time: RealArray, state: RealArray) -> RealArray:
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
        algo_name="Radau",
        rtol=1e-6,
        atol=1e-6,
    )

    assert allclose(
        ode_problem.result.state_trajectories[0],
        discipline_result["position_0"],
        atol=5e-2,
    )

    assert allclose(
        ode_problem.result.state_trajectories[2],
        discipline_result["position_1"],
        atol=5e-2,
    )


def test_chained_masses_wrong_size_parameters():
    """Test the two methods to reproduce the movement of three masses connected by
    springs, by performing an MDA on coupled ODEDisciplines and by defining a single
    ODEDiscipline whose dynamic is the result of an MDA."""
    masses = [1.0, 2.0, 3.0]
    stiffnesses = [1.0, 1.0, 1.0]

    with pytest.raises(ValueError) as err:
        CoupledSpringsGenerator(masses=masses, stiffnesses=stiffnesses)
    assert "Incompatible lengths of 'masses' and 'stiffnesses'" in str(err.value)


def test_lateral_masses_as_floats():
    """Test the case when the position of the lateral masses are initialized as
    floats."""

    times = linspace(0.0, 10, 30)
    ode_discipline_1 = SpringODEDiscipline(
        mass=1,
        left_stiffness=1,
        right_stiffness=1,
        times=times,
        state_names=("position_1", "velocity_1"),
        state_dot_names=("position_1_dot", "velocity_1_dot"),
        left_position=0.0,
        right_position=0.0,
        left_position_name="left_position",
        right_position_name="right_position",
        is_left_position_fixed=True,
        is_right_position_fixed=True,
        rtol=1e-12,
        atol=1e-12,
    )
    ode_discipline_1.execute()


def test_one_mass_attached_to_moving_pins():
    """Test the case where a single mass is linked to mobile pins to its
    left and right, moving according to a given trajectory."""
    mass = 1.0
    left_stiff = 1.0
    right_stiff = 1.0
    omega = 1.0
    times = linspace(0, 2.0, 30)

    def fct_left_position(t: RealArray | float = 0.0):
        return sin(omega * t)

    def fct_right_position(t: RealArray | float = 0.0):
        return sin(omega * t)

    left_position = fct_left_position(times)
    right_position = fct_right_position(times)

    ode_discipline = SpringODEDiscipline(
        mass=mass,
        left_stiffness=left_stiff,
        right_stiffness=right_stiff,
        left_position=left_position,
        right_position=right_position,
        is_left_position_fixed=True,
        is_right_position_fixed=True,
        times=times,
        state_names=("position", "velocity"),
        rtol=1e-12,
        atol=1e-12,
    )

    discipline_result = ode_discipline.execute({
        "position": array([0.0]),
        "velocity": array([0.0]),
    })

    def _ode_func(time: RealArray, state: RealArray) -> RealArray:
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
        algo_name="Radau",
        rtol=1e-12,
        atol=1e-12,
    )

    assert allclose(
        ode_problem.result.state_trajectories[0],
        discipline_result["position"],
        atol=1e-3,
    )


def test_one_mass_attached_to_mobile_pins_wrong_time_lengths():
    """Test the error messages when the positions provided for the
    left and right masses are incompatible with times."""
    mass = 1.0
    left_stiff = 1.0
    right_stiff = 1.0
    omega = 1.0
    times = linspace(0, 2, 30)
    other_times = linspace(0, 2, 20)

    def fct_left_position(t: RealArray | float = 0.0):
        return sin(omega * t)

    def fct_right_position(t: RealArray | float = 0.0):
        return sin(omega * t)

    left_position = fct_left_position(other_times)
    right_position = fct_right_position(times)

    ode_discipline_1 = SpringODEDiscipline(
        mass=mass,
        left_stiffness=left_stiff,
        right_stiffness=right_stiff,
        left_position=left_position,
        right_position=right_position,
        is_left_position_fixed=True,
        is_right_position_fixed=True,
        times=times,
        state_names=("position", "velocity"),
        rtol=1e-12,
        atol=1e-12,
    )

    with pytest.raises(AssertionError) as err:
        ode_discipline_1.execute()
    assert "Incoherent lengths" in str(err.value)

    left_position = fct_left_position(times)
    right_position = fct_right_position(other_times)

    ode_discipline_2 = SpringODEDiscipline(
        mass=mass,
        left_stiffness=left_stiff,
        right_stiffness=right_stiff,
        left_position=left_position,
        right_position=right_position,
        is_left_position_fixed=True,
        is_right_position_fixed=True,
        times=times,
        state_names=("position", "velocity"),
        rtol=1e-12,
        atol=1e-12,
    )

    with pytest.raises(AssertionError) as err:
        ode_discipline_2.execute()
    assert "Incoherent lengths" in str(err.value)
