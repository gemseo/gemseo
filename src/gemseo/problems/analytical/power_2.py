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
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
A quadratic analytical problem
******************************
"""
from __future__ import annotations

import logging

from numpy import array
from numpy import ndarray
from numpy import sum as np_sum

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction

LOGGER = logging.getLogger(__name__)


class Power2(OptimizationProblem):
    """**Power2** is a very basic quadratic analytical :class:`.OptimizationProblem`:

    - Objective to minimize: :math:`x_0^2 + x_1^2 + x_2^2`
    - Inequality constraint 1: :math:`x_0^3 - 0.5 > 0`
    - Inequality constraint 2: :math:`x_1^3 - 0.5 > 0`
    - Equality constraint: :math:`x_2^3 - 0.9 = 0`
    - Analytical optimum: :math:`x^*=(0.5^{1/3}, 0.5^{1/3}, 0.9^{1/3})`
    """

    def __init__(
        self, exception_error: bool = False, initial_value: float = 1.0
    ) -> None:
        """
        Args:
            exception_error: Whether to raise an error when calling the objective;
                useful for tests.
            initial_value: The initial design value of the problem.
        """
        design_space = DesignSpace()
        design_space.add_variable("x", 3, l_b=-1.0, u_b=1.0, value=initial_value)

        super().__init__(design_space)
        self.objective = MDOFunction(
            self.pow2,
            name="pow2",
            f_type="obj",
            jac=self.pow2_jac,
            expr="x[0]**2 + x[1]**2 + x[2]**2",
            args=["x"],
        )
        self.add_ineq_constraint(
            MDOFunction(
                self.ineq_constraint1,
                name="ineq1",
                f_type="ineq",
                jac=self.ineq_constraint1_jac,
                expr="0.5 - x[0]**3",
                args=["x"],
            )
        )

        self.add_ineq_constraint(
            MDOFunction(
                self.ineq_constraint2,
                name="ineq2",
                f_type="ineq",
                jac=self.ineq_constraint2_jac,
                expr="0.5 - x[1]**3",
                args=["x"],
            )
        )

        self.add_eq_constraint(
            MDOFunction(
                self.eq_constraint,
                name="eq",
                f_type="eq",
                jac=self.eq_constraint_jac,
                expr="0.9 - x[2]**3",
                args=["x"],
            )
        )
        self.iter_error = 0
        self.exception_error = exception_error

    def pow2(self, x_dv: ndarray) -> ndarray:
        """Compute the objective :math:`x_0^2 + x_1^2 + x_2^2`.

        Args:
            x_dv: The design variable vector.

        Returns:
            The objective value.

        Raises:
            ValueError: When :attr:`.exception_error` is ``True``
                and the method has already been called three times.
        """
        if self.exception_error:
            if self.iter_error >= 3:
                raise ValueError("pow2() has already been called three times.")

            self.iter_error += 1

        return np_sum(x_dv**2)

    @staticmethod
    def pow2_jac(x_dv: ndarray) -> ndarray:
        """Compute the gradient of the objective.

        Args:
            x_dv: The design variable vector.

        Returns:
            The value of the objective gradient.
        """
        return 2 * x_dv

    @staticmethod
    def ineq_constraint1(x_dv: ndarray) -> ndarray:
        """Compute the first inequality constraint :math:`x_0^3 - 0.5 > 0`.

        Args:
            x_dv: The design variable vector.

        Returns:
            The value of the first inequality constraint.
        """
        return -x_dv[[0]] ** 3 + 0.5

    @staticmethod
    def ineq_constraint2(x_dv: ndarray) -> ndarray:
        """Compute the second inequality constraint :math:`x_1^3 - 0.5 > 0`.

        Args:
            x_dv: The design variable vector.

        Returns:
            The value of the second inequality constraint.
        """
        return -x_dv[[1]] ** 3 + 0.5

    @staticmethod
    def eq_constraint(x_dv: ndarray) -> ndarray:
        """Compute the equality constraint :math:`x_2^3 - 0.9 = 0`.

        Args:
            x_dv: The design variable vector.

        Returns:
            The value of the equality constraint.
        """
        return -x_dv[[2]] ** 3 + 0.9

    @staticmethod
    def ineq_constraint1_jac(x_dv: ndarray) -> ndarray:
        """Compute the gradient of the first inequality constraint.

        Args:
            x_dv: The design variable vector.

        Returns:
            The value of the gradient of the first inequality constraint.
        """
        return -array([3 * x_dv[0] * x_dv[0], 0.0, 0.0])

    @staticmethod
    def ineq_constraint2_jac(x_dv: ndarray) -> ndarray:
        """Compute the gradient of the second inequality constraint.

        Args:
            x_dv: The design variable vector.

        Returns:
            The value of the gradient of the second inequality constraint.
        """
        return -array([0, 3 * x_dv[1] * x_dv[1], 0.0])

    @staticmethod
    def eq_constraint_jac(x_dv: ndarray) -> ndarray:
        """Compute the gradient of the equality constraint.

        Args:
            x_dv: The design variable vector.

        Returns:
            The value of the gradient of the equality constraint.
        """
        return -array([0.0, 0.0, 3 * x_dv[2] * x_dv[2]])

    @staticmethod
    def get_solution() -> tuple[ndarray, ndarray]:
        """Return the analytical solution of the problem.

        Returns:
            The theoretical optimum of the problem.
        """
        x_opt = array([0.5 ** (1.0 / 3.0), 0.5 ** (1.0 / 3.0), 0.9 ** (1.0 / 3.0)])
        f_opt = np_sum(x_opt**2)
        return x_opt, f_opt
