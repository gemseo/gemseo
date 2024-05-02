# Copyright 2022 Airbus SAS
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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Gabriel Max DE MENDONÇA ABRANTES
r"""Viennet multi-objective problem.

See :cite:`viennet1996multicriteria`.

.. math::

   \begin{aligned}
   \text{minimize the objective function}
   & f_1(x, y) = (x^2 + y^2) / 2 + sin(x^2 + y^2) \\
   & f_2(x, y) = (3x - 2y + 4)^2 / 8 + (x - y + 1)^2 / 27 + 15 \\
   & f_3(x, y) = 1 / (x^2 + y^2 + 1) - 1.1 e^{-(x^2 + y^2)} \\
   \text{with respect to the design variables}&x,\,y \\
   \text{subject to the bound constraints}
   & -3.0 \leq x \leq 3.0\\
   & -3.0 \leq y \leq 3.0
   \end{aligned}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import cos as np_cos
from numpy import exp as np_exp
from numpy import sin as np_sin

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import RealArray


class Viennet(OptimizationProblem):
    """The Viennet optimization problem."""

    def __init__(
        self,
        l_b: float = -3.0,
        u_b: float = 3.0,
        initial_guess: Iterable[float] = (0, 0),
    ) -> None:
        """
        Args:
            l_b: The lower bound (common value to all variables).
            u_b: The upper bound (common value to all variables).
            initial_guess: The initial guess for the optimal solution.
        """  # noqa: D205 D212
        design_space = DesignSpace()
        design_space.add_variable("x", l_b=l_b, u_b=u_b)
        design_space.add_variable("y", l_b=l_b, u_b=u_b)
        design_space.set_current_value(array(initial_guess))
        super().__init__(design_space)

        # Set the objective function.
        self.objective = MDOFunction(
            self._compute_output,
            name=self.__class__.__name__,
            f_type=MDOFunction.FunctionType.OBJ,
            jac=self._compute_jacobian,
            expr="[(x**2 + y**2) / 2 + sin(x**2 + y**2), 9*x - (y-1)**2,"
            "(3*x - 2*y + 4)**2 / 8 + (x - y + 1)^2 / 27 + 15,"
            "1 / (x**2 + y**2 + 1) - 1.1*exp(-(x**2 + y**2))]",
            input_names=["x", "y"],
            dim=3,
        )

    @staticmethod
    def _compute_output(design_variables: RealArray) -> RealArray:
        """Compute the objectives of the Viennet function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The objective function value.
        """
        x, y = design_variables
        xy2 = x**2 + y**2

        return array([
            0.5 * xy2 + np_sin(xy2),
            (3.0 * x - 2 * y + 4.0) ** 2 / 8.0 + (x - y + 1.0) ** 2 / 27.0 + 15.0,
            1.0 / (xy2 + 1.0) - 1.1 * np_exp(-xy2),
        ])

    @staticmethod
    def _compute_jacobian(design_variables: RealArray) -> RealArray:
        """Compute the gradient of the objective function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The gradient of the objective functions w.r.t. the design variables.
        """
        x, y = design_variables
        xy2 = x**2 + y**2

        return array([
            [x + 2.0 * x * np_cos(xy2), y + 2.0 * y * np_cos(xy2)],
            [
                3.0 * (3.0 * x - 2 * y + 4.0) / 4.0 + 2 * (x - y + 1.0) / 27.0,
                -2.0 * (3.0 * x - 2 * y + 4.0) / 4.0 - 2 * (x - y + 1.0) / 27.0,
            ],
            [
                -2.0 * x * (xy2 + 1.0) ** (-2) + 1.1 * 2 * x * np_exp(-xy2),
                -2.0 * y * (xy2 + 1.0) ** (-2) + 1.1 * 2 * y * np_exp(-xy2),
            ],
        ])
