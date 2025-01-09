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
r"""The 2-body astrodynamics problem.

Predict the motion and position of a massive object
orbiting a fixed mass
in an elliptic trajectory with a given eccentricity.
This problem is treated here as a classical central force problem.

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
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & 1 \\
        \frac{2x^2 - y^2}{(x^2 + y^2)^{5/2}} & \frac{3xy}{(x^2 + y^2)^{5/2}} & 0 & 0 \\
        \frac{3xy}{(x^2 + y^2)^{5/2}} & \frac{-x^2 + 2y^2}{(x^2 + y^2)^{5/2}} & 0 & 0
    \end{pmatrix}.
"""

from __future__ import annotations

from math import pi
from math import sqrt
from typing import TYPE_CHECKING

from numpy import arctan
from numpy import array
from numpy import cos
from numpy import floor
from numpy import sin
from numpy import tan
from numpy import zeros
from scipy.optimize import fsolve

from gemseo.algos.ode.ode_problem import ODEProblem

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.typing import RealArray

_default_times = array([(0, 0.5)])


def _compute_rhs(
    time=0.0,
    position_x=0.5,
    position_y=0.0,
    velocity_x=1.0,
    velocity_y=3.0,
):
    """Compute the right-hand side function of the ODE.

    Args:
        time: The time.
        position_x: The horizontal coordinate of the mass at ``time``.
        position_y: The vertical coordinate of the mass at ``time``.
        velocity_x: The horizontal component of the velocity at ``time``.
        velocity_y: The vertical component of the velocity at ``time``.

    Returns:
        The horizontal velocity of the mass at ``time``,
        the vertical velocity of the mass at ``time``,
        the horizontal acceleration of the mass at ``time``
        and the vertical acceleration of the mass at ``time``.
    """
    den = sqrt(position_x**2 + position_y**2) ** 3
    position_x_dot = velocity_x
    position_y_dot = velocity_y
    velocity_x_dot = -position_x / den
    velocity_y_dot = -position_y / den
    return position_x_dot, position_y_dot, velocity_x_dot, velocity_y_dot  # noqa: RET504


def _compute_rhs_jacobian(
    time: float = 0.0,
    position_x: float = 0.5,
    position_y: float = 0.0,
    velocity_x: float = 1.0,
    velocity_y: float = 3.0,
) -> RealArray:
    """Compute the Jacobian of the right-hand side of the ODE.

    Args:
        time: The time.
        position_x: The horizontal coordinate of the mass at ``time``.
        position_y: The vertical coordinate of the mass at ``time``.
        velocity_x: The horizontal component of the velocity at ``time``.
        velocity_y: The vertical component of the velocity at ``time``.

    Returns:
        The Jacobian of the function describing the dynamic of the system
        with respect to the state variables,
        evaluated at ``time``.
    """
    den = sqrt(position_x**2 + position_y**2) ** 5
    jac = zeros((4, 4))
    jac[0, 2] = jac[1, 3] = 1
    jac[2, 0] = (2 * position_x**2 - position_y**2) / den
    jac[3, 1] = (-(position_x**2) + 2 * position_y**2) / den
    jac[2, 1] = jac[3, 0] = (3 * position_x * position_y) / den
    return jac


class KeplerEquationSolver:
    """A class to solve the Kepler equation."""

    __eccentricity: float
    """The eccentricity of the particle trajectory."""

    __mean_anomaly: float
    """The mean anomaly."""

    def __init__(self, eccentricity: float) -> None:
        """
        Args:
            eccentricity: The eccentricity of the particle trajectory.
        """  # noqa: D205 D212
        self.__eccentricity = eccentricity
        self.__mean_anomaly = 0.0

    def __compute_residual_kepler(self, e: float) -> float:
        """Compute the Kepler equation in the residual form.

        Args:
             e: The eccentric anomaly.

        Returns:
              The residual of the Kepler equation.
        """
        return e - self.__eccentricity * sin(e) - self.__mean_anomaly

    def __call__(
        self, mean_anomaly: float
    ) -> tuple[RealArray, Mapping[str, RealArray], int, str]:
        """Solve the Kepler equation.

        Args:
            mean_anomaly: The mean anomaly.

        Returns:
            The result of the root finding algorithm.
        """
        self.__mean_anomaly = mean_anomaly
        return fsolve(self.__compute_residual_kepler, x0=array([mean_anomaly]))


class OrbitalDynamics(ODEProblem):
    r"""Problem describing a particle following an elliptical orbit.

    The particle is supposed to
    represent a massive object immersed in a gravitational field
    generated by a single, fixed, massive object.

    The dynamic is parametrized by its eccentricity,
    a strictly positive parameter,
    that is supposed to be between 0 (included),
    and 1 (excluded) for elliptic orbits.
    All parameters of the system are rescaled
    so that the length of the semi-major axis of the trajectory is normalized to 1.

    By Kepler's law,
    the orbit of the particle is planar and follows a conic section.
    Thus,
    the reference system is oriented so that the trajectory lies in the x-y plane,
    and the periapsis is placed on the axis y=0.
    The horizontal coordinate of the periapsis can be computed as x = 1 - eccentricity.
    The time interval is scaled so that the mass reaches its periapsis in t=0.
    At the periapsis,
    the velocity of the massive particle is orthogonal
    to the line of the apsides,
    and its magnitude is equal to
    :math:`sqrt{frac{1 + eccentricity}{1 - eccentricity}}`.
    We suppose that the orbit follows a counter-clockwise path.

    The state of the ODE consists in four variables:

    - :math:`x`, the horizontal coordinate,
    - :math:`y`, the vertical coordinate,
    - :math:`v_x`, the horizontal component of the velocity,
    - :math:`v_y`, the vertical component of the velocity.

    The system follows the dynamic below:

    - :math:`\frac{dx}{dt} = v_x`,
    - :math:`\frac{dy}{dt} = v_y`,
    - :math:`\frac{dv_x}{dt} = - x / (\sqrt(x^2 + y^2))^3`,
    - :math:`\frac{dv_y}{dt} = - y / (\sqrt(x^2 + y^2))^3`.
    """

    __eccentricity: float
    """The eccentricity of the particle trajectory."""

    def __init__(
        self,
        eccentricity: float = 0.5,
        times: RealArray = (0.0, 0.5),
    ) -> None:
        """
        Args:
            eccentricity: The eccentricity of the particle trajectory.
            times: The initial and final time.
        """  # noqa: D205 D212
        initial_state = array([
            1 - eccentricity,
            0,
            0,
            sqrt((1 + eccentricity) / (1 - eccentricity)),
        ])
        self.__eccentricity = eccentricity
        self.__kepler_equation_solver = KeplerEquationSolver(eccentricity)
        super().__init__(
            func=self._func,
            jac_function_wrt_state=self._jac_wrt_state,
            initial_state=initial_state,
            times=times,
        )

    @staticmethod
    def _func(time: float, state: RealArray) -> RealArray:
        x, y, vx, vy = state.T
        return array(_compute_rhs(time, x, y, vx, vy))

    @staticmethod
    def _jac_wrt_state(time: float, state: RealArray) -> RealArray:
        x, y, vx, vy = state.T
        return array(_compute_rhs_jacobian(time, x, y, vx, vy))

    def compute_analytic_solution(self, times: RealArray):
        """Compute the analytic solution of the Kepler problem for an elliptic orbit.

        Args:
            times: The times_eval.

        Returns:
            The x- and y- positions of the mass.
        """
        eccentric_anomaly = array([
            self.__kepler_equation_solver(time) for time in times
        ]).ravel()
        period = 2 * pi
        true_anomaly = 2 * arctan(
            sqrt((1 + self.__eccentricity) / (1 - self.__eccentricity))
            * abs(tan(eccentric_anomaly / 2))
        )
        true_anomaly += (floor(times * 2 / period) % 2 == 1) * (
            period - 2 * true_anomaly
        )
        radii_vector = 1 - self.__eccentricity * cos(eccentric_anomaly)
        x_positions = radii_vector * cos(true_anomaly)
        y_positions = radii_vector * sin(true_anomaly)
        return x_positions, y_positions
