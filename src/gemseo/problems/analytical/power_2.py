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
A quadratic analytical problem
******************************
"""
from __future__ import division, unicode_literals

import logging

from numpy import array
from numpy import sum as np_sum

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction

LOGGER = logging.getLogger(__name__)


class Power2(OptimizationProblem):
    """**Power2** is a very basic quadratic analytical
    :class:`.OptimizationProblem`:

    - Objective to minimize: :math:`x_{dv,0}^2+x_1^2+x_2^2`
    - Inequality constraint 1: :math:`x_{dv,0}^3 - 0.5 > 0`
    - Inequality constraint 2: :math:`x_{dv,1}^3 - 0.5 > 0`
    - Equality constraint: :math:`x_{dv,2}^3 - 0.9 = 0`
    - Analytical optimum: :math:`(0.5^{1/3}, 0.5^{1/3}, 0.9^{1/3})`
    """

    def __init__(self, exception_error=False, initial_value=1.0):
        """The constructor initializes the Power2 :class:`.OptimizationProblem` by
        defining the :class:`.DesignSpace`, the objective function and the constraints.

        :param exception_error: if True, call to the objective raises errors
            useful for tests
        :type exception_error: bool
        """
        design_space = DesignSpace()
        design_space.add_variable("x", 3, l_b=-1.0, u_b=1.0, value=initial_value)

        super(Power2, self).__init__(design_space)
        self.objective = MDOFunction(
            self.pow2,
            name="pow2",
            f_type="obj",
            jac=self.pow2_jac,
            expr="x[0]**2+x[1]**2+x[2]**2",
            args=["x"],
        )
        ineq1 = MDOFunction(
            self.ineq_constraint1,
            name="ineq1",
            f_type="ineq",
            jac=self.ineq_constraint1_jac,
            expr="0.5 -x[0] ** 3",
            args=["x"],
        )
        self.add_ineq_constraint(ineq1)

        ineq2 = MDOFunction(
            self.ineq_constraint2,
            name="ineq2",
            f_type="ineq",
            jac=self.ineq_constraint2_jac,
            expr="0.5 -x[1] ** 3",
            args=["x"],
        )
        self.add_ineq_constraint(ineq2)

        eq_c = MDOFunction(
            self.eq_constraint,
            name="eq",
            f_type="eq",
            jac=self.eq_constraint_jac,
            expr="x[2] ** 3 - 0.9",
            args=["x"],
        )
        self.add_eq_constraint(eq_c)
        self.iter_error = 0
        self.exception_error = exception_error

    def pow2(self, x_dv):
        """Compute the objective of analytical function.

        :param x_dv: design variable vector
        :type x_dv: numpy array
        :returns: objective value
        :rtype: float
        """
        if self.exception_error:
            if self.iter_error < 3:
                self.iter_error += 1
                obj = np_sum(x_dv ** 2)
            else:
                raise ValueError
        else:
            obj = np_sum(x_dv ** 2)
        return obj

    @staticmethod
    def pow2_jac(x_dv):
        """Compute the gradient of objective.

        :param x_dv: design variable vector
        :type x_dv: numpy array
        :returns: value of the objective gradient
        :rtype: numpy array
        """
        return 2 * x_dv

    @staticmethod
    def ineq_constraint1(x_dv):
        """Compute the first inequality constraint.

        :param x_dv: design variable vector
        :type x_dv: numpy array
        :returns: value of the first inequality constraint
        :rtype: numpy array
        """
        return -array([(x_dv[0] ** 3 - 0.5)])

    @staticmethod
    def ineq_constraint2(x_dv):
        """Compute the second inequality constraint.

        :param x_dv: design variable vector
        :type x_dv: numpy array
        :returns: value of the second inequality constraint
        :rtype: numpy array
        """
        return -array([(x_dv[1] ** 3 - 0.5)])

    @staticmethod
    def eq_constraint(x_dv):
        """Compute the equality constraint.

        :param x_dv: design variable vector
        :type x_dv: numpy array
        :returns: value of the equality constraint
        :rtype: numpy array
        """
        return -array([x_dv[2] ** 3 - 0.9])

    @staticmethod
    def ineq_constraint1_jac(x_dv):
        """Compute the first inequality constraint gradient.

        :param x_dv: design variable vector
        :type x_dv: numpy array
        :returns: value of the first inequality constraint gradient
        :rtype: numpy array
        """
        return -array([3 * x_dv[0] * x_dv[0], 0.0, 0.0])

    @staticmethod
    def ineq_constraint2_jac(x_dv):
        """Compute the second inequality constraint gradient.

        :param x_dv: design variable vector
        :type x_dv: numpy array
        :returns: value of the second inequality constraint gradient
        :rtype: numpy array
        """
        return -array([0, 3 * x_dv[1] * x_dv[1], 0.0])

    @staticmethod
    def eq_constraint_jac(x_dv):
        """Compute the equality constraint gradient.

        :param x_dv: design variable vector
        :type x_dv: numpy array
        :returns: value of the equality constraint gradient
        :rtype: numpy array
        """
        return -array([0.0, 0.0, 3 * x_dv[2] * x_dv[2]])

    @staticmethod
    def get_solution():
        """Return analytical result of optimization.

        :returns: theoretical optimum
        :rtype: numpy array
        """
        x_opt = array([0.5 ** (1.0 / 3.0), 0.5 ** (1.0 / 3.0), 0.9 ** (1.0 / 3.0)])
        f_opt = Power2().pow2(x_opt)
        return x_opt, f_opt
