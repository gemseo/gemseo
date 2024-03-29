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
#
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author: Vincent DROUET
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Sub-optimization constraint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import atleast_2d
from numpy import zeros

if TYPE_CHECKING:
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from gemseo.typing import NumberArray


class SubOptimConstraint:
    r"""The constraint of the sub-problem.

    This is used to compute
    :math:`f(x) - \phi^T\beta - t * n`
    without using a closure.
    """

    __f: MDOFunction
    """The objective function."""

    __n: NumberArray
    """The quasi-normal vector to the phi simplex."""

    __phi_beta: NumberArray
    r"""The scalar product of :math:`\phi` and :math:`beta`."""

    def __init__(self, phi_beta: NumberArray, n: NumberArray, f: MDOFunction) -> None:
        """
        Args:
            phi_beta: The scalar product of phi and beta.
            n: The quasi-normal vector to the phi simplex.
            f: The objective function.
        """  # noqa: D205, D212, D415
        self.__phi_beta = phi_beta
        self.__n = n
        self.__f = f

    def compute_output(self, x_t: NumberArray) -> NumberArray:
        """Compute the constraint function output at :math:`x`.

        Args:
            x_t: A vector :math:`x` followed by a scalar :math:`t`.

        Returns:
            The constraint function output at :math:`x`.
        """
        return self.__f.evaluate(x_t[:-1]) - self.__phi_beta - x_t[-1] * self.__n

    def compute_jacobian(self, x_t: NumberArray) -> NumberArray:
        """Compute the constraint function Jacobian at :math:`x`.

        Args:
            x_t: A vector :math:`x` followed by a scalar :math:`t`.

        Returns:
            The constraint function Jacobian at :math:`x`.
        """
        jac_f = atleast_2d(self.__f.jac(x_t[:-1]))
        jac = zeros((jac_f.shape[0], jac_f.shape[1] + 1))
        jac[:, :-1] = jac_f
        jac[:, -1] = -self.__n
        return jac
