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
r"""Poloni's bi-objective optimization problem.

See :cite:`POLONI2000403`.

.. math::

    \begin{aligned}
    &a1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)\\
    &a2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)\\
    &b1(x, y) = 0.5 * sin(x) - 2 * cos(x) + sin(y) - 1.5 * cos(y)\\
    &b2(x, y) = 1.5 * sin(x) - cos(x) + 2 * sin(y) - 0.5 * cos(y)\\
    \text{minimize the objective function}\\
    & f_1(x, y) = 1 + (a1 - b1(x,y)^2 + (a2 - b2(x,y))^2 \\
    & f_2(x, y) = (x + 3)^2 + (y + 1)^2 \\
    \text{with respect to the design variables}\\
    &x \\
    \text{subject to the bound constraints}\\
    & -\pi \leq x \leq \pi\\
    & -\pi \leq y \leq \pi
    \end{aligned}
"""

from __future__ import annotations

from math import cos
from math import pi
from math import sin
from typing import TYPE_CHECKING

from numpy import array

from gemseo import create_design_space
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Poloni(OptimizationProblem):
    """Poloni multi-objective, bound constrained optimization problem."""

    def __init__(self) -> None:  # noqa: D205 D212 D107
        design_space = create_design_space()
        design_space.add_variable("x", lower_bound=-pi, upper_bound=pi, value=0)
        design_space.add_variable("y", lower_bound=-pi, upper_bound=pi, value=0)
        super().__init__(design_space)
        self.objective = MDOFunction(
            self._compute_output, self.__class__.__name__, jac=self._compute_jacobian
        )

    @staticmethod
    def _compute_output(x: RealArray) -> RealArray:
        """Compute the output of the function.

        Args:
            x: The values to compute the output of the function.

        Returns:
            The output of the function.
        """
        x, y = x
        a1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)
        a2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)
        b1 = 0.5 * sin(x) - 2 * cos(x) + sin(y) - 1.5 * cos(y)
        b2 = 1.5 * sin(x) - cos(x) + 2 * sin(y) - 0.5 * cos(y)
        f2 = 1 + (a1 - b1) ** 2 + (a2 - b2) ** 2
        f1 = (x + 3) ** 2 + (y + 1) ** 2
        return array([f1, f2])

    @staticmethod
    def _compute_jacobian(x: RealArray) -> RealArray:
        """Compute the Jacobian of the function.

        Args:
            x: The values to compute the Jacobian of the function.

        Returns:
            The Jacobian value of the function.
        """
        x, y = x
        a1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)
        a2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)
        b1 = 0.5 * sin(x) - 2 * cos(x) + sin(y) - 1.5 * cos(y)
        b2 = 1.5 * sin(x) - cos(x) + 2 * sin(y) - 0.5 * cos(y)

        amb1 = a1 - b1
        amb2 = a2 - b2
        df2_dx = -amb1 * (cos(x) + 4 * sin(x)) - amb2 * (3 * cos(x) + 2 * sin(x))
        df2_dy = -amb1 * (2 * cos(y) + sin(x)) - amb2 * (4 * cos(y) + sin(y))
        df1_dx = 2 * (x + 3)
        df1_dy = 2 * (y + 1)
        return array([[df1_dx, df1_dy], [df2_dx, df2_dy]])
