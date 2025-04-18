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
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Binh and Korn multi-objective problem.

This module implements the Binh and Korn multi-objective problem as defined in
:cite:`binh1997mobes`:

.. math::

   \begin{aligned}
   \text{minimize the objective function } & f_1(x, y) = 4x^2 + 4y^2 \\
   & f_2(x, y) = (x-5)^2 + (y-5)^2 \\
   \text{with respect to the design variables }&x,\,y \\
   \text{subject to the general constraints }
   & g_1(x,y) = (x-5)^2 + y^2 \leq 25.0\\
   & g_2(x, y) = (x-8)^2 + (y+3)^2 \geq 7.7\\
   \text{subject to the bound constraints }
   & 0 \leq x \leq 5.0\\
   & 0 \leq y \leq 3.0
   \end{aligned}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import ndarray
from numpy import zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.utils.matplotlib_figure import FigSizeType


class BinhKorn(OptimizationProblem):
    """Binh and Korn optimization problem.

    The constructor initializes the BinhKorn :class:`.OptimizationProblem` by defining
    the :class:`.DesignSpace`, the objective function and the constraints.
    """

    def __init__(self, initial_values: FigSizeType = (1.0, 1.0)) -> None:
        """
        Args:
            initial_values: Initial value of the design variables.
        """  # noqa: D205 D212
        design_space = DesignSpace()
        design_space.add_variable(
            "x", lower_bound=0.0, upper_bound=5.0, value=initial_values[0]
        )
        design_space.add_variable(
            "y", lower_bound=0.0, upper_bound=3.0, value=initial_values[1]
        )

        super().__init__(design_space)
        self.objective = MDOFunction(
            self._compute_binhkorn,
            name="compute_binhkorn",
            f_type="obj",
            jac=self._compute_binhkorn_jac,
            expr="(4*x**2+ 4*y**2, (x-5.)**2 + (y-5.)**2)",
            input_names=["x", "y"],
            dim=2,
        )
        ineq1 = MDOFunction(
            self._compute_ineq_constraint1,
            name="ineq1",
            f_type=MDOFunction.ConstraintType.INEQ,
            jac=self._compute_ineq_constraint1_jac,
            expr="(x-5.)**2 + y**2 <= 25.",
            input_names=["x", "y"],
        )
        self.add_constraint(ineq1)

        ineq2 = MDOFunction(
            self._compute_ineq_constraint2,
            name="ineq2",
            f_type=MDOFunction.ConstraintType.INEQ,
            jac=self._compute_ineq_constraint2_jac,
            expr="(x-8.)**2 + (y+3)**2 >= 7.7",
            input_names=["x", "y"],
        )
        self.add_constraint(ineq2)

    @staticmethod
    def _compute_binhkorn(
        x_dv: ndarray,
    ) -> ndarray:
        """Compute the objective of the BinhKorn problem.

        Args:
            x_dv: The design variable vector.

        Returns:
            The objective function value.
        """
        obj = array([0.0, 0.0])
        obj[0] = 4 * x_dv[0] ** 2 + 4 * x_dv[1] ** 2
        obj[1] = (x_dv[0] - 5.0) ** 2 + (x_dv[1] - 5.0) ** 2
        return obj

    @staticmethod
    def _compute_ineq_constraint1(
        x_dv: ndarray,
    ) -> ndarray:
        """Compute the first constraint.

        Args:
            x_dv: The design variable vector.

        Returns:
            The first constraint value.
        """
        return array([(x_dv[0] - 5.0) ** 2 + x_dv[1] - 25.0])

    @staticmethod
    def _compute_ineq_constraint2(
        x_dv: ndarray,
    ) -> ndarray:
        """Compute the second constraint.

        Args:
            x_dv: The design variable vector.

        Returns:
            The second constraint value.
        """
        return array([-((x_dv[0] - 8.0) ** 2) - (x_dv[1] + 3) + 7.7])

    @staticmethod
    def _compute_binhkorn_jac(
        x_dv: ndarray,
    ) -> ndarray:
        """Compute the gradient of the objective.

        Args:
            x_dv: The design variables vector.

        Returns:
            The gradient of the objective functions
            w.r.t the design variables
        """
        jac = zeros([2, 2])
        jac[0, 0] = 8.0 * x_dv[0]
        jac[0, 1] = 8.0 * x_dv[1]
        jac[1, 0] = 2.0 * x_dv[0] - 10.0
        jac[1, 1] = 2.0 * x_dv[1] - 10.0
        return jac

    @staticmethod
    def _compute_ineq_constraint1_jac(
        x_dv: ndarray,
    ):  # (...) -> ndarray
        """Compute the first inequality constraint Jacobian.

        Args:
            x_dv: The design variables vector.

        Returns:
            The gradient of the first constraint function
            w.r.t the design variables.
        """
        jac = zeros([1, 2])
        jac[0, 0] = 2.0 * x_dv[0] - 10.0
        jac[0, 1] = 2.0 * x_dv[1]
        return jac

    @staticmethod
    def _compute_ineq_constraint2_jac(
        x_dv: ndarray,
    ):  # (...) -> ndarray
        """Compute the second inequality constraint Jacobian.

        Args:
            x_dv: The design variables vector.

        Returns:
            The gradient of the second constraint function
            w.r.t the design variables.
        """
        jac = zeros([1, 2])
        jac[0, 0] = -2.0 * x_dv[0] + 16.0
        jac[0, 1] = -2.0 * x_dv[1] + 6.0
        return jac
