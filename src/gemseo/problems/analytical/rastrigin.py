# -*- coding: utf-8 -*-
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
The Rastrigin analytic problem
******************************
"""
from __future__ import division, unicode_literals

import logging
from cmath import cos, pi, sin

from numpy import array, ones, zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction

LOGGER = logging.getLogger(__name__)


class Rastrigin(OptimizationProblem):

    r"""**Rastrigin** :class:`.OptimizationProblem`
    uses the Rastrigin objective function
    with the :class:`.DesignSpace` :math:`[-0.1,0.1]^2`

    From http://en.wikipedia.org/wiki/Rastrigin_function:

        the Rastrigin function is a non-convex function used as a
        performance test problem for optimization algorithms.
        It is a typical example of non-linear multimodal function.
        It was first proposed by [Rastrigin] as a 2-dimensional
        function and has been generalized by [MuhlenbeinEtAl].
        Finding the minimum of this function is a fairly difficult
        problem due to its large search space and its large
        number of local minima.
        It has a global minimum at :math:`x=0` where :math:`f(x)=0`.
        It can be extended to :math:`n>2` dimensions:

        .. math::

           f(x) = 10n + \sum_{i=1}^n [x_i^2 - 10\cos(2\pi x_i)]

        [Rastrigin] Rastrigin, L. A. "Systems of extremal control."
        Mir, Moscow (1974).

        [MuhlenbeinEtAl] H. Mühlenbein, D. Schomisch and J. Born.
        "The Parallel Genetic Algorithm as Function Optimizer ".
        Parallel Computing, 17, pages 619–632, 1991.
    """

    def __init__(self):
        """
        The constructor initializes the Rastrigin
        :class:`.OptimizationProblem`
        by defining the :class:`.DesignSpace`
        and the objective function.
        """
        design_space = DesignSpace()
        design_space.add_variable("x", 2, l_b=-0.1, u_b=0.1)
        design_space.set_current_x(0.01 * ones(2))
        super(Rastrigin, self).__init__(design_space)
        expr = "20 + sum(x[i]**2 - 10*cos(2pi*x[i]))"
        self.objective = MDOFunction(
            self.rastrigin,
            name="Rastrigin",
            f_type="obj",
            jac=self.rastrigin_jac,
            expr=expr,
            args=["x"],
        )

    @staticmethod
    def rastrigin(x_dv):
        """This function computes the order n=2 Rastrigin function.

        :param x_dv: design variable vector of size 2
        :returns: result of Rastrigin function evaluation
        """
        a_c = 10.0
        func = (
            a_c * 2.0
            + (x_dv[0] ** 2 - a_c * cos(2 * pi * x_dv[0]))
            + (x_dv[1] ** 2 - a_c * cos(2 * pi * x_dv[1]))
        )
        return func.real

    @staticmethod
    def get_solution():
        """Return theoretical optimal value of Rastrigin function.

        :returns: design variables values of optimized values,
            function value at optimum
        :rtype: numpy array
        """
        x_opt = zeros(2)
        f_opt = 0.0
        return x_opt, f_opt

    @staticmethod
    def rastrigin_jac(x_dv):
        """This function computes the analytical gradient of 2nd order Rastrigin
        function.

        :param x_dv: design variable vector
        :type x_dv: numpy array
        :returns: analytical gradient vector of Rastrigin function
        :rtype: numpy array
        """
        a_c = 10.0
        analytic_grad = array(
            [
                2 * x_dv[0] + 2 * pi * a_c * sin(2 * pi * x_dv[0]),
                2 * x_dv[1] + 2 * pi * a_c * sin(2 * pi * x_dv[1]),
            ]
        )
        return analytic_grad.real
