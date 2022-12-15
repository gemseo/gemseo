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
"""An analytical problem to test the non-early termination of optimization algorithms."""
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.python_compatibility import Final
from numpy import array
from numpy import ndarray
from numpy import zeros


class Constant(OptimizationProblem):
    """A Toy analytical :class:`.OptimizationProblem`.

    It is currently used to test the premature termination of some optimization algorithms,
    when the criterion n_stop_crit_x is not properly set (see bug #307).

    The objective to minimize is :math:`x`.
    """

    CONSTANT: Final[ndarray] = array([1.0])

    def __init__(self, initial_value: float = 1.0) -> None:
        """
        Args:
            initial_value: The initial design value of the problem.
        """
        design_space = DesignSpace()
        design_space.add_variable("x", l_b=-1.0, u_b=1.0, value=initial_value)

        super().__init__(design_space)
        self.objective = MDOFunction(
            self.__compute_constant,
            name="constant",
            f_type="obj",
            jac=self.__compute_constant_jac,
            args=["x"],
        )

    def __compute_constant(self, x_dv: ndarray) -> ndarray:
        """Compute the objective :math:`x`.

        Args:
            x_dv: The design variable vector.

        Returns:
            The objective value.
        """
        return self.CONSTANT

    @staticmethod
    def __compute_constant_jac(x_dv: ndarray) -> ndarray:
        """Compute the gradient of the objective function.

        Args:
            x_dv: The design variable vector.

        Returns:
            The value of the objective gradient.
        """
        return zeros(len(x_dv))
