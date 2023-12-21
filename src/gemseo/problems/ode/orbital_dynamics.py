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
# Copyright 2023 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
r"""The 2-body astrodynamics problem.

Predict the motion and position of two massive objects viewed as point particles
that only interact with one another using classical mechanics. This problem is
treated here as a classical central force problem.

Consider the frame defined by one particle. The position :math:`(x, y)` and the
velocity :math:`(v_x, v_y)` of the other particle as a function of time can be described
by the following set of equations:

.. math:

    \left\{ \begin{align}
        \dot{x(t)}   &= v_x(t) \\
        \dot{y(t)}   &= v_y(t) \\
        \dot{v_x(t)} &= - \frac{x(t)}{r^3} \\
        \dot{v_y(t)} &= - \frac{y(t)}{r^3} \\
    \end{align}\right,

with :math:`r = \sqrt{x(t)^2 + y(t)^2}`.

We use the initial conditions:

.. math:

    \left\{ \begin{align}
        x(0)   &= 1 - e \\
        y(0)   &= 0 \\
        v_x(0) &= 0 \\
        v_y(0) &= \sqrt{\frac{1+e}{1-e}} \\
    \end{align}\right,

where :math:`e` is the eccentricity of the particle trajectory.

The Jacobian of the right-hand side of this ODE is:

.. math:

    \begin{pmatrix}
        0 & 0 & \frac{2x^2 - y^2}{(x^2 + y^2)^{5/2}}
            & \frac{3xy}{(x^2 + y^2)^{5/2}} \\
        0 & 0 & \frac{3xy}{(x^2 + y^2)^{5/2}}
            & \frac{-x^2 + 2y^2}{(x^2 + y^2)^{5/2}} \\
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0
    \end{pmatrix}.
"""

from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

from numpy import array
from numpy import zeros

from gemseo.algos.ode.ode_problem import ODEProblem

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _compute_rhs(time: float, state: NDArray[float]) -> NDArray[float]:  # noqa:U100
    """Compute the right-hand side of ODE."""
    x, y, vx, vy = state
    r = sqrt(x * x + y * y)
    f1 = vx
    f2 = vy
    f3 = -x / r / r / r
    f4 = -y / r / r / r
    return array([f1, f2, f3, f4])


def _compute_rhs_jacobian(time: float, state: NDArray[float]) -> NDArray[float]:  # noqa:U100
    """Compute the Jacobian of the right-hand side of the ODE."""
    x, y, _, _ = state
    jac = zeros((4, 4))
    jac[0, 2] = (2 * x * x - y * y) / (x * x + y * y) ** (5 / 2)
    jac[0, 3] = (3 * x * y) / (x * x + y * y) ** (5 / 2)
    jac[1, 2] = (3 * x * y) / (x * x + y * y) ** (5 / 2)
    jac[1, 3] = (-x * x + 2 * y * y) / (x * x + y * y) ** (5 / 2)
    jac[2, 0] = 1
    jac[3, 1] = 1
    return jac


class OrbitalDynamics(ODEProblem):
    """Equations of motion of a massive point particle under a central force."""

    def __init__(
        self,
        eccentricity: float = 0.5,
        use_jacobian: bool = True,
        state_vector: NDArray[float] | None = None,
    ) -> None:
        r"""
        Args:
            eccentricity: The eccentricity of the particle trajectory.
            use_jacobian: Whether to use the analytical expression of the Jacobian.
            state_vector: The state vector
                :math:`s(t)=(x(t), y(t), \dot{x(t)}, \dot{y(t)})`
                of the system.
        """  # noqa: D205 D212
        if state_vector is None:
            state_vector = zeros(2)
        self.state_vector = state_vector

        initial_state = array([
            1 - eccentricity,
            0,
            0,
            sqrt((1 + eccentricity) / (1 - eccentricity)),
        ])

        jac = _compute_rhs_jacobian if use_jacobian else None
        super().__init__(
            func=_compute_rhs,
            jac=jac,
            initial_state=initial_state,
            initial_time=0,
            final_time=0.5,
        )
