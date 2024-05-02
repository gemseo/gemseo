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
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author:  François Gallard - minor improvements for integration
r"""Fonseca-Fleming bi-objective optimization problem.

See :cite:`fonseca1995overview`.

.. math::

   \begin{aligned}
   \text{minimize the objective function}
   & f_1(x) = 1 - exp(-\sum_{i=1}^{d}((x_i - 1 / sqrt(d)) ^ 2)) \\
   & f_2(x) = 1 + exp(-\sum_{i=1}^{d}((x_i + 1 / sqrt(d)) ^ 2)) \\
   \text{with respect to the design variables}&x\\
   \text{subject to the bound constraints}
   & x\in[-4,4]^d
   \end{aligned}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import exp
from numpy import full
from numpy import sqrt
from numpy import square
from numpy import sum as np_sum
from numpy import vstack
from numpy import zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class FonsecaFleming(OptimizationProblem):
    """Fonseca-Fleming multi-objective, bound constrained optimization problem."""

    __a: int
    """The inverse square root of the design vector size."""

    def __init__(self, dimension: int = 3) -> None:
        """
        Args:
            dimension: The design vector size.
        """  # noqa: D205 D212
        self.__a = 1 / sqrt(dimension)
        design_space = DesignSpace()
        design_space.add_variable(
            "x",
            size=dimension,
            l_b=full(dimension, -4.0),
            u_b=full(dimension, 4.0),
            value=zeros(dimension),
        )
        super().__init__(design_space)
        self.objective = MDOFunction(
            self._compute_output, self.__class__.__name__, jac=self._compute_jacobian
        )

    def _compute_output(self, x: RealArray) -> RealArray:
        """Compute the output of the function.

        Args:
            x: The values to compute the output of the function.

        Returns:
            The output of the function.
        """
        return 1 - exp(-np_sum(square([x - self.__a, x + self.__a]), axis=1))

    def _compute_jacobian(self, x: RealArray) -> RealArray:
        """Compute the Jacobian of the function.

        Args:
            x: The values to compute the Jacobian of the function.

        Returns:
            The Jacobian value of the function.
        """
        f1_j = 2 * (x - self.__a) * exp(-np_sum((x - self.__a) ** 2))
        f2_j = 2 * (x + self.__a) * exp(-np_sum((x + self.__a) ** 2))
        return vstack([f1_j, f2_j])
