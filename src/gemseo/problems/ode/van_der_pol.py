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
r"""The Van der Pol (VDP) problem describing an oscillator with non-linear damping.

Van der Pol, B. & Van Der Mark, J.
Frequency Demultiplication.
Nature 120, 363-364 (1927).

The Van der Pol problem is written as follows:

.. math::

    \frac{d^2 x(t)}{dt^2} -
    \mu (1-x(t)^2) \frac{dx(t)}{dt} + x = 0

where :math:`x(t)` is the position coordinate as a function of time, and
:math:`\mu` is a scalar parameter indicating the stiffness.

This problem can be rewrittent in a 2-dimensional form with only first-order
derivatives. Let :math:`y = \frac{dx}{dt}` and
:math:`s = \begin{pmatrix}x\\y\end{pmatrix}`. Then the Van der Pol problem is:

.. math::

    \frac{ds}{dt} = f(s, t)

with

.. math::

    f : s = \begin{pmatrix} x \\ y \end{pmatrix} \mapsto
    \begin{pmatrix} y \\ \mu (1-x^2) y - x \end{pmatrix}

The jacobian of this function can be expressed analytically:

.. math::

    \mathrm{Jac}\, f = \begin{pmatrix}
        0 & 1 \\
        -2\mu xy - 1 & \mu (1 - x^2)
    \end{pmatrix}

There is no exact solution to the Van der Pol oscillator problem in terms of
known tabulated functions (see Panayotounakos *et al.*, On the Lack of Analytic
Solutions of the Van Der Pol Oscillator. ZAMM 83, nᵒ 9 (1 septembre 2003)).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import zeros

from gemseo.algos.ode.ode_problem import ODEProblem

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VanDerPol(ODEProblem):
    """Representation of an oscillator with non-linear damping."""

    _mu: float
    r"""Stiffness parameter.

    Van der Pol is stiffer with larger values of :math:`\mu`.
    """

    state_vect: NDArray[float]
    r"""State vector :math:`s=(x, \dot{x})` of the system."""

    def __init__(
        self,
        initial_time: float = 0,
        final_time: float = 0.5,
        mu: float = 1.0e3,
        use_jacobian: bool = True,
        state_vector: NDArray[float] = None,
    ) -> None:
        """
        Args:
            mu: The stiffness parameter.
            initial_time: The start of the integration interval.
            final_time: The end of the integration interval.
            use_jacobian: Whether to use the analytical expression of the Jacobian.
                If false, use finite differences to estimate the Jacobian.
            state_vector: The state vector of the system.
        """  # noqa: D205 D212
        self._mu = mu

        if state_vector is None:
            state_vector = zeros(2)
        self.state_vect = state_vector

        initial_state = array([
            2 + self.state_vect[0],
            -2 / 3
            + 10 / (81 * self._mu)
            - 292 / (2187 * self._mu * self._mu)
            + self.state_vect[1],
        ])

        jac = self.__compute_rhs_jacobian if use_jacobian else None
        super().__init__(
            func=self.__compute_rhs,
            jac=jac,
            initial_state=initial_state,
            initial_time=initial_time,
            final_time=final_time,
        )

    def __compute_rhs(self, time: float, state: NDArray[float]) -> NDArray[float]:  # noqa:U100
        """Compute the right-hand side of the ODE.

        Args:
            time: The time at which the function is evaluated.
            state: The state in which the function is evaluated.

        Returns:
             The value of the right-hand side of the ODE.
        """
        return array([state[1], self._mu * state[1] * (1 - state[0] ** 2) - state[0]])

    def __compute_rhs_jacobian(
        self, time: float, state: NDArray[float]
    ) -> NDArray[float]:  # noqa:U100
        """Compute the Jacobian of the right-hand side of the ODE.

        Args:
            time: The time at which the function is evaluated.
            state: The state in which the function is evaluated.

        Returns:
             The Jacobian of the right-hand side of the ODE.
        """
        jac = zeros((2, 2))
        jac[1, 0] = -self._mu * 2 * state[1] * state[0] - 1
        jac[0, 1] = 1
        jac[1, 1] = self._mu * (1 - state[0] * state[0])
        return jac
