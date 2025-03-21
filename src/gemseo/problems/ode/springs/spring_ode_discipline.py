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

r"""The discipline for describing the motion of a single mass connected by springs."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from numpy import asarray
from numpy import cos
from numpy import sin
from numpy import sqrt

from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.problems.ode.springs.springs_dynamics_discipline import STATE_DOT_NAMES
from gemseo.problems.ode.springs.springs_dynamics_discipline import STATE_NAMES
from gemseo.problems.ode.springs.springs_dynamics_discipline import (
    SpringsDynamicsDiscipline,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.typing import RealArray


class SpringODEDiscipline(ODEDiscipline):
    """A discipline representing a mass connected by two springs."""

    def __init__(
        self,
        mass: float,
        left_stiffness: float,
        right_stiffness: float,
        times: RealArray,
        left_position: float = 0.0,
        right_position: float = 0.0,
        state_names: Sequence[str] = STATE_NAMES,
        state_dot_names: Sequence[str] = STATE_DOT_NAMES,
        left_position_name: str = "left_position",
        right_position_name: str = "right_position",
        is_left_position_fixed: bool = False,
        is_right_position_fixed: bool = False,
        **ode_solver_options: Any,
    ) -> None:
        """
        Args:
            mass: The value of the mass.
            left_stiffness: The stiffness of the spring on the left-hand side.
            right_stiffness: The stiffness of the spring on the right-hand side.
            left_position: The position of the other extremity
                of the spring to the left.
            right_position: The position of the other extremity
                of the spring to the right.
            times: The times at which the solution must be evaluated.
            state_names: The names of the state variables.
            state_dot_names: The names of the derivatives of the state variables
                relative to time.
            left_position_name: The names of the variable describing the motion
                of the mass to the left.
            right_position_name: The names of the variable describing the motion
                of the mass to the right.
            is_left_position_fixed: Whether the other end of the spring
                to the left is fixed.
            is_right_position_fixed: Whether the other end of the spring
                to the right is fixed.
            **ode_solver_options: The options of the ODE solver.
        """  # noqa: D205, D212, D415
        if not is_left_position_fixed:
            left_position = asarray(left_position) * len(times)
        if not is_right_position_fixed:
            right_position = asarray(right_position) * len(times)

        mass_spring_discipline = SpringsDynamicsDiscipline(
            mass=mass,
            left_stiffness=left_stiffness,
            right_stiffness=right_stiffness,
            left_position=left_position,
            right_position=right_position,
            is_left_position_fixed=is_left_position_fixed,
            is_right_position_fixed=is_right_position_fixed,
            times=times,
            state_names=state_names,
            state_dot_var_names=state_dot_names,
            left_position_name=left_position_name,
            right_position_name=right_position_name,
        )

        super().__init__(
            rhs_discipline=mass_spring_discipline,
            state_names=dict(zip(state_names, state_dot_names)),
            times=times,
            return_trajectories=True,
            **ode_solver_options,
        )

    @staticmethod
    def compute_analytic_mass_position(
        initial_position: float,
        initial_velocity: float,
        left_stiffness: float,
        right_stiffness: float,
        mass: float,
        times: RealArray,
    ) -> RealArray:
        r"""Compute the analytic position of the mass :math:`m`.

        The equation describing the motion of the mass is

        .. math::

           \left\{ \begin{cases}
               \dot{x} &= y \\
               \dot{y} &= \frac{k_1 + k_2}{m} x
           \end{cases} \right.

        where :math:`k_1` and :math:`k_2` are the stiffnesses of the springs.

        If :math:`x(t=0) = x_0` and :math:`y(t=0) = y_0`, then the general expression
        for the position :math:`x(t)` at time :math:`t` is

        .. math::

            x(t) = \frac{1}{2}\big( x + \frac{y_0}{\omega} \big) \exp^{i\omega t}
                + \frac{1}{2}\big( x - \frac{y_0}{\omega} \big) \exp^{-i\omega t}

        with :math:`\omega = \frac{-k_1+k_2}{m}`.

        Args:
            initial_position: The initial position.
            initial_velocity: The initial velocity.
            left_stiffness: The stiffness of the spring to the left.
            right_stiffness: The stiffness of the spring to the right.
            mass: The value of the mass.
            times: The time(s) at which the position of the mass should be evaluated.

        Returns:
            The position(s) of the mass.
        """
        omega = sqrt((right_stiffness + left_stiffness) / mass)
        omega_time = omega * times
        return (initial_position * cos(omega_time)) + (
            initial_velocity / omega * sin(omega_time)
        )
