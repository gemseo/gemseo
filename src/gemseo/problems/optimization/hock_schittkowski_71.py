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

r"""Hock & Schittkowski problem 71.

This module implements the Hock & Schittkowski non-linear programming problem 71.

See: Willi Hock and Klaus Schittkowski. (1981) Test Examples for
Nonlinear Programming Codes. Lecture Notes in Economics and Mathematical
Systems Vol. 187, Springer-Verlag.
Based on MATLAB code by Peter Carbonetto.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class HockSchittkowski71(OptimizationProblem):
    """Hock and Schittkowski problem 71."""

    def __init__(
        self,
        initial_guess: NumberArray = (1.0, 5.0, 5.0, 1.0),
    ):
        """Initialize the Hock Schittkowski 71 problem.

        Args:
            initial_guess: The initial guess for the optimal solution.
        """
        design_space = DesignSpace()
        design_space.add_variable(
            "x",
            size=4,
            lower_bound=[1.0, 1.0, 1.0, 1.0],
            upper_bound=[5.0, 5.0, 5.0, 5.0],
        )

        design_space.set_current_value(array(initial_guess))

        super().__init__(design_space)

        self.objective = MDOFunction(
            self.compute_objective,
            name="hock_schittkoski_71",
            f_type=MDOFunction.FunctionType.OBJ,
            jac=self.compute_objective_jacobian,
            expr="x_1 * x_4 * (x_1 + x_2 + x_3) + x_3",
            input_names=["x"],
            dim=1,
        )

        equality_constraint = MDOFunction(
            self.compute_equality_constraint,
            name="equality_constraint",
            f_type=MDOFunction.ConstraintType.EQ,
            jac=self.compute_equality_constraint_jacobian,
            expr="(x_1**2 + x_2**2 + x_3**2 + x_4**2) - 40",
            input_names=["x"],
            dim=1,
        )
        self.add_constraint(
            equality_constraint, constraint_type=MDOFunction.ConstraintType.EQ
        )

        inequality_constraint = MDOFunction(
            self.compute_inequality_constraint,
            name="inequality_constraint",
            f_type=MDOFunction.ConstraintType.INEQ,
            jac=self.compute_inequality_constraint_jacobian,
            expr="25 - (x_1 * x_2 * x_3 * x_4)",
            input_names=["x"],
            dim=1,
        )
        self.add_constraint(
            inequality_constraint, constraint_type=MDOFunction.ConstraintType.INEQ
        )

    @staticmethod
    def compute_objective(design_variables: NumberArray) -> NumberArray:
        """Compute the objectives of the Hock and Schittkowski 71 function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The objective function value.
        """
        return (
            design_variables[0] * design_variables[3] * (sum(design_variables[0:3]))
            + design_variables[2]
        )

    @staticmethod
    def compute_objective_jacobian(design_variables: NumberArray) -> NumberArray:
        """Compute the Jacobian of objective function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The gradient of the objective functions wrt the design variables.
        """
        x_1, x_2, x_3, x_4 = design_variables

        return array([
            x_1 * x_4 + x_4 * (x_1 + x_2 + x_3),
            x_1 * x_4,
            x_1 * x_4 + 1.0,
            x_1 * (x_1 + x_2 + x_3),
        ])

    @staticmethod
    def compute_equality_constraint(design_variables: NumberArray) -> NumberArray:
        """Compute the equality constraint function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The equality constraint's value.
        """
        x_1, x_2, x_3, x_4 = design_variables
        return array([x_1**2 + x_2**2 + x_3**2 + x_4**2 - 40])

    @staticmethod
    def compute_inequality_constraint(design_variables: NumberArray) -> NumberArray:
        """Compute the inequality constraint function.

        Args:
            design_variables: The design variables vector.

        Returns:
            The inequality constraint's value.
        """
        x_1, x_2, x_3, x_4 = design_variables
        return array([25 - x_1 * x_2 * x_3 * x_4])

    @staticmethod
    def compute_equality_constraint_jacobian(
        design_variables: NumberArray,
    ) -> NumberArray:
        """Compute the equality constraint's Jacobian.

        Args:
            design_variables: The design variables vector.

        Returns:
            The Jacobian of the equality constraint function wrt the design variables.
        """
        x_1, x_2, x_3, x_4 = design_variables

        return array([
            2 * x_1,
            2 * x_2,
            2 * x_3,
            2 * x_4,
        ])

    @staticmethod
    def compute_inequality_constraint_jacobian(
        design_variables: NumberArray,
    ) -> NumberArray:
        """Compute the inequality constraint's Jacobian.

        Args:
            design_variables: The design variables vector.

        Returns:
            The Jacobian of the inequality constraint function wrt the design variables.
        """
        x_1, x_2, x_3, x_4 = design_variables

        return array([
            -x_2 * x_3 * x_4,
            -x_1 * x_3 * x_4,
            -x_1 * x_2 * x_4,
            -x_1 * x_2 * x_3,
        ])

    @staticmethod
    def get_solution() -> tuple[NumberArray, NumberArray]:
        """Return the analytical solution of the problem.

        Returns:
            The theoretical optimum of the problem.
        """
        x_opt = array([0.99999999, 4.74299964, 3.82114998, 1.37940829])
        f_opt = x_opt[0] * x_opt[3] * sum(x_opt[0:3]) + x_opt[2]
        return x_opt, f_opt
