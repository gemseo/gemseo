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
"""The Rosenbrock analytic problem."""

from __future__ import annotations

from numpy import ndarray
from numpy import ones
from numpy import zeros
from scipy.optimize import rosen
from scipy.optimize import rosen_der

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction


class Rosenbrock(OptimizationProblem):
    r"""The Rosenbrock optimization problem.

    .. math::

       f(x) = \sum_{i=2}^{n_x} 100(x_{i} - x_{i-1}^2)^2 + (1 - x_{i-1})^2

    with the default :class:`.DesignSpace`
    :math:`[-0.2,0.2]^{n_x}`.
    """

    def __init__(
        self,
        n_x: int = 2,
        l_b: float = -2.0,
        u_b: float = 2.0,
        scalar_var: bool = False,
        initial_guess: ndarray | None = None,
    ) -> None:
        """
        Args:
            n_x: The dimension of the design space.
            l_b: The lower bound (common value to all variables).
            u_b: The upper bound (common value to all variables).
            scalar_var: If ``True``,
                the design space will contain only scalar variables
                (as many as the problem dimension);
                if ``False``,
                the design space will contain a single multidimensional variable
                (whose size equals the problem dimension).
            initial_guess: The initial guess for optimal solution.
        """  # noqa: D205 D212
        design_space = DesignSpace()
        if scalar_var:
            args = [f"x{i}" for i in range(1, n_x + 1)]
            for arg in args:
                design_space.add_variable(arg, lower_bound=l_b, upper_bound=u_b)
        else:
            args = ["x"]
            design_space.add_variable("x", size=n_x, lower_bound=l_b, upper_bound=u_b)
        if initial_guess is None:
            design_space.set_current_value(zeros(n_x))
        else:
            design_space.set_current_value(initial_guess)

        super().__init__(design_space)
        self.objective = MDOFunction(
            rosen,
            name="rosen",
            jac=rosen_der,
            expr="sum( 100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2 )",
            input_names=args,
        )

    def get_solution(self) -> tuple[ndarray, float]:
        """Return the theoretical optimal value.

        Returns:
            The design variables and the objective at optimum.
        """
        return ones(self.design_space.dimension), 0.0
