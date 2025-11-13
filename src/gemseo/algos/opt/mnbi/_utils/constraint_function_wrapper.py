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
"""Constraint function wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import atleast_2d
from numpy import zeros

if TYPE_CHECKING:
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.typing import NumberArray


class ConstraintFunctionWrapper:
    """Wrapper of the constraint function for the beta sub-optimization.

    In the main optimization,
    the constraint $g$ requires the vector $x$,
    but in the beta sub-optimization,
    they are executed with input $[x, t]$ where $t$ is a scalar.
    This wrapper removes the $t$ component
    and calls $g$ with only the $x$ input.
    """

    __g: MDOFunction
    """The constraint function to be wrapped."""

    def __init__(self, g: MDOFunction) -> None:
        """
        Args:
            g: The constraint function $g$ to be wrapped.
        """  # noqa: D205, D212, D415
        self.__g = g

    def compute_output(self, x_t: NumberArray) -> NumberArray:
        """Compute the constraint function output at $x$.

        Args:
            x_t: A vector $x$ followed by a scalar $t$.

        Returns:
            The constraint function output at $x$.
        """
        return self.__g.evaluate(x_t[:-1])

    def compute_jacobian(self, x_t: NumberArray) -> NumberArray:
        """Compute the constraint function Jacobian at $x$.

        Args:
            x_t: A vector $x$ followed by a scalar $t$.

        Returns:
            The constraint function Jacobian at $x$.
        """
        jac_g = atleast_2d(self.__g.jac(x_t[:-1]))
        jac = zeros((jac_g.shape[0], jac_g.shape[1] + 1))
        jac[:, :-1] = jac_g
        return jac
