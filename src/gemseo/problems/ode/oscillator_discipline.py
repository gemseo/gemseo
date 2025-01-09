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
r"""A harmonic oscillator.

In classical mechanics,
the position :math:`x` of a harmonic oscillator is described by the equation

.. math::

    \frac{d^2x}{dt^2} = -\omega^2 x


with :math:`\omega \in \mathbb{R}_+^*`.
This second-order Ordinary Differential Equation (ODE)
has an analytical solution:

.. math::

    x(t) = \lambda \sin(\omega t) + \mu \cos(\omega t)


where :math:`\lambda` and :math:`\mu` are two constants
defined by the initial conditions.

This solution can be re-written
as a two-dimensional first-order ODE:

.. math::

    \begin{cases}
    \frac{dx}{dt} = v, \\
    \frac{dv}{dt} = -\omega^2 x.
    \end{cases}

where :math:`v` represents the velocity of the oscillator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import ndarray

from gemseo.core.discipline.base_discipline import CacheType
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.typing import RealArray


_time = array([0.0])
_position = array([0.0])
_velocity = array([1.0])


class OscillatorDiscipline(ODEDiscipline):
    """A discipline representing a harmonic oscillator."""

    __omega_squared: float
    """The squared angular velocity of the oscillator."""

    def __init__(
        self,
        omega: float,
        times: RealArray,
        return_trajectories: bool = False,
        final_state_names: Mapping[str, str] = READ_ONLY_EMPTY_DICT,
        cache_inner_discipline_is_none=True,
    ):
        """
        Args:
            omega: The positive angular velocity of the oscillator.
        """  # noqa: D205, D212, D415
        self.__omega_squared = omega**2
        rhs_discipline = AutoPyDiscipline(py_func=self._compute_rhs)
        if cache_inner_discipline_is_none:
            rhs_discipline.set_cache(cache_type=CacheType.NONE)

        super().__init__(
            times=times,
            state_names=("position", "velocity"),
            rhs_discipline=rhs_discipline,
            return_trajectories=return_trajectories,
            final_state_names=final_state_names,
            rtol=1e-12,
            atol=1e-12,
        )

    def _compute_rhs(
        self,
        time: ndarray = _time,
        position: ndarray = _position,
        velocity: ndarray = _velocity,
    ) -> tuple[ndarray, ndarray]:
        """Compute the right-hand side of the ODE equation.

        Args:
            time: The value of the time.
            position: The position of the system at ``time``.
            velocity: The velocity of the system at ``time``.

        Returns:
            The derivative of the position at ``time``.
            The derivative of the velocity at ``time``.
        """
        position_dot = velocity
        velocity_dot = -self.__omega_squared * position
        return position_dot, velocity_dot  # noqa: RET504
