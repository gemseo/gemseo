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
#    INITIAL AUTHORS - initial API and implementation and/or
#    initial documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
# from Tedford2010
# Propane combustion, p12
r"""
The propane combustion MDO problem
**********************************

The Propane MDO problem can be found in :cite:`Padula1996`
and :cite:`TedfordMartins2006`. It represents the chemical equilibrium
reached during the combustion of propane in air.
Variables are assigned to represent each of the ten combustion products
as well as the sum of the products.

The optimization problem is as follows:


.. math::

   \begin{aligned}
   \text{minimize the objective function } &
   f_2 + f_6 + f_7 + f_9 \\
   \text{with respect to the design variables } &
   x_{1},\,x_{3},\,x_{6},\,x_{7} \\
   \text{subject to the general constraints }
   & f_2(x) \geq 0\\
   & f_6(x) \geq 0\\
   & f_7(x) \geq 0\\
   & f_9(x) \geq 0\\
   \text{subject to the bound constraints }
   & x_{1} \geq 0\\
   & x_{3} \geq 0\\
   & x_{6} \geq 0\\
   & x_{7} \geq 0\\
   \end{aligned}

where the System Discipline consists of computing the following expressions:

.. math::

   \begin{aligned}
   f_2(x) & = & 2x_1 + x_2 + x_4 + x_7 + x_8 + x_9 + 2x_{10} - R, \\
   f_6(x) & = & K_6x_2^{1/2}x_4^{1/2} - x_1^{1/2}x_6(p/x_{11})^{1/2}, \\
   f_7(x) & = & K_7x_1^{1/2}x_2^{1/2} - x_4^{1/2}x_7(p/x_{11})^{1/2}, \\
   f_9(x) & = & K_9x_1x_3^{1/2} - x_4x_9(p/x_{11})^{1/2}. \\
   \end{aligned}


Discipline 1 computes :math:`(x_{2}, x_{4})`
by satisfying the following equations:

.. math::

   \begin{aligned}
   x_1 + x_4 - 3 &=& 0,\\
   K_5x_2x_4 - x_1x_5 &=& 0.\\
   \end{aligned}

Discipline 2 computes :math:`(x_2, x_4)` such that:

.. math::

   \begin{aligned}
   K_8x_1 + x_4x_8(p/x_{11}) &=& 0,\\
   K_{10}x_{1}^{2} - x_4^2x_{10}(p/x_{11}) &=& 0.\\
   \end{aligned}

and Discipline 3 computes :math:`(x_5, x_9, x_{11})` by solving:

.. math::

   \begin{aligned}
   2x_2 + 2x_5 + x_6 + x_7 - 8&=& 0,\\
   2x_3 + x_9 - 4R &=& 0, \\
   x_{11} - \sum_{j=1}^{10} x_j &=& 0. \\
   \end{aligned}
"""
from __future__ import division, unicode_literals

from cmath import sqrt
from os.path import dirname, join

from numpy import array, complex128, ones, zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline


def get_design_space(to_complex=True):
    """Reads the design space file.

    :param to_complex: if True, current x is a complex vector
    """
    ds_read = DesignSpace.read_from_txt(
        join(dirname(__file__), "propane_design_space.txt")
    )
    if to_complex:
        ds_read.to_complex()
    return ds_read


class PropaneReaction(MDODiscipline):

    """Propane's objective and constraints discipline This discipline's outputs are the
    objective function and partial terms used in inequality constraints.

    Note: the equations have been decoupled (y_i = y_i(x_shared)). Otherwise,
    the solvers may find iterates for
    which discipline analyses are not computable.
    """

    def __init__(self):
        super(PropaneReaction, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = {
            "x_shared": ones(4, dtype=complex128),
            "y_1": ones(2, dtype=complex128),
            "y_2": ones(2, dtype=complex128),
            "y_3": ones(3, dtype=complex128),
        }
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _run(self):
        """Compute the outputs of the propane combustion model."""
        inputs = ["y_1", "y_2", "y_3", "x_shared"]
        y_1, y_2, y_3, x_shared = self.get_inputs_by_name(inputs)
        f2_list = [self.f_2(x_shared, y_1, y_2, y_3)]
        f_2 = array(f2_list, dtype=complex128)
        f_6 = array([self.f_6(x_shared, y_1, y_3)], dtype=complex128)
        f_7 = array([self.f_7(x_shared, y_1, y_3)], dtype=complex128)
        f_9 = array([self.f_9(x_shared, y_1, y_3)], dtype=complex128)
        obj = f_2 + f_7 + f_6 + f_9
        # constraints on f_2, f_6, f_7, f_9 must be nonnegative
        # in original problem don't forget to take the opposite
        self.store_local_data(f_2=-f_2, f_6=-f_6, f_7=-f_7, f_9=-f_9, obj=obj)

    @classmethod
    def f_2(cls, x_shared, y_1, y_2, y_3):
        """First term of a sum of four in the objective function. Is also a nonnegative
        constraint at system level.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :param y_1: first coupling variable
        :type y_1: ndarray
        :param y_2: second coupling variable
        :type y_2: ndarray
        :param y_3: third coupling variable
        :type y_3: ndarray
        :returns: f2(x_shared, y_1, y_2, y_3)
        :rtype: float
        """
        return (
            2.0 * x_shared[0]
            + y_1[0]
            + y_1[1]
            + x_shared[3]
            + y_2[1]
            + y_3[0]
            + 2.0 * y_3[1]
            - 10.0
        )

    @classmethod
    def f_6(cls, x_shared, y_1, y_3):
        """Second term of a sum of four in the objective function. Is also a nonnegative
        constraint at system level.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :param y_1: first coupling variable
        :type y_1: ndarray
        :param y_3: third coupling variable
        :type y_3: ndarray
        :returns: f6(x, y)
        :rtype: float
        """
        return sqrt(y_1[0] * y_1[1]) - sqrt(40.0 * x_shared[0] / y_3[2]) * x_shared[2]

    @classmethod
    def f_7(cls, x_shared, y_1, y_3):
        """Third term of a sum of four in the objective function. Is also a nonnegative
        constraint at system level.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :param y_1: first coupling variable
        :type y_1: ndarray
        :param y_3: third coupling variable
        :type y_3: ndarray
        :returns: f7(x, y)
        :rtype: float
        """
        return sqrt(x_shared[0] * y_1[0]) - sqrt(40.0 * y_1[1] / y_3[2]) * x_shared[3]

    @classmethod
    def f_9(cls, x_shared, y_1, y_3):
        """Fourth term of a sum of four in the objective function. Is also a nonnegative
        constraint at system level.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :param y_1: first coupling variable
        :type y_1: ndarray
        :param y_3: third coupling variable
        :type y_3: ndarray
        :returns: f9(x, y)
        :rtype: float
        """
        return x_shared[0] * sqrt(x_shared[1]) - y_1[1] * y_3[0] * sqrt(40.0 / y_3[2])


class PropaneComb1(MDODiscipline):

    """Propane combustion 1st set of equations This discipline is characterized by two
    governing equations."""

    def __init__(self):
        super(PropaneComb1, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = {"x_shared": ones(4, dtype=complex128)}
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _run(self):
        """Solve 2 coupling equations in functional form and compute coupling variables
        Y0 and y_1."""
        x_shared = self.get_inputs_by_name("x_shared")
        y_1_out = zeros(2, dtype=complex128)
        y_1_out[0] = self.compute_y0(x_shared)
        y_1_out[1] = self.compute_y1(x_shared)
        self.store_local_data(y_1=y_1_out)

    @classmethod
    def compute_y0(cls, x_shared):
        """Solve the first coupling equation in functional form.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :returns: coupling variable y0
        """
        return -x_shared[0] * (x_shared[2] + x_shared[3] - 8.0) / 6.0

    @classmethod
    def compute_y1(cls, x_shared):
        """Solve the second coupling equation in functional form.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :returns: coupling variable y1
        """
        return 3.0 - x_shared[0]


class PropaneComb2(MDODiscipline):
    """Propane combustion 2nd set of equations This discipline is characterized by two
    governing equations."""

    def __init__(self):
        super(PropaneComb2, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = {"x_shared": ones(4, dtype=complex128)}
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _run(self):
        """Solve 2 coupling equations in functional form and compute coupling variables
        y_2 and y_3."""
        x_shared = self.get_inputs_by_name("x_shared")
        y_2_out = zeros(2, dtype=complex128)
        y_2_out[0] = self.compute_y2(x_shared)
        y_2_out[1] = self.compute_y3(x_shared)
        self.store_local_data(y_2=y_2_out)

    @classmethod
    def compute_y2(cls, x_shared):
        """Solve the third coupling equation in functional form.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :returns: coupling variable y_2
        """
        return (x_shared[0] - 3.0) * (x_shared[2] + x_shared[3] - 8.0) / 6.0

    @classmethod
    def compute_y3(cls, x_shared):
        """Solve the fourth coupling equation in functional form.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :returns: coupling variable y_3
        """
        y_3 = -(x_shared[0] - 3.0) * x_shared[0]
        y_3 *= x_shared[2] + x_shared[3] - 2.0 * x_shared[1] + 94.0
        y_3 /= 2.0 * (400.0 * x_shared[0] ** 2.0 - 2403.0 * x_shared[0] + 3600.0)
        return y_3


class PropaneComb3(MDODiscipline):

    """This discipline is characterized by three governing equations."""

    def __init__(self):
        super(PropaneComb3, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = {"x_shared": ones(4, dtype=complex128)}
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _run(self):
        """Solve 3 coupling equations in functional form and compute coupling variables
        y_4, Y5 and Y6."""
        x_shared = self.get_inputs_by_name("x_shared")
        y_3_out = zeros(3, dtype=complex128)
        y_3_out[0] = self.compute_y4(x_shared)
        y_3_out[1] = self.compute_y5(x_shared)
        y_3_out[2] = self.compute_y6(x_shared)
        self.store_local_data(y_3=y_3_out)

    @classmethod
    def compute_y4(cls, x_shared):
        """Solve the fifth coupling equation in functional form.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :returns: coupling variable y_4
        """
        return 40.0 - 2.0 * x_shared[1]

    @classmethod
    def compute_y5(cls, x_shared):
        """Solve the sixth coupling equation in functional form.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :returns: coupling variable Y5
        """
        y_5 = x_shared[0] ** 2
        y_5 *= x_shared[2] + x_shared[3] - 2.0 * x_shared[1] + 94.0
        y_5 /= 2.0 * (400.0 * x_shared[0] ** 2 - 2403.0 * x_shared[0] + 3600.0)
        return y_5

    @classmethod
    def compute_y6(cls, x_shared):
        """Solve the seventh coupling equation in functional form.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :returns: coupling variable Y6
        """
        y_6 = 200.0 * (x_shared[0] - 3.0) ** 2
        y_6 *= x_shared[2] + x_shared[3] - 2.0 * x_shared[1] + 94.0
        y_6 /= 400.0 * x_shared[0] ** 2 - 2403.0 * x_shared[0] + 3600.0
        return y_6


# =========================================================================
# Analytical Jacobians of f and residuals #
# =========================================================================
#
# def jacobian_f_x(x_shared, Y):
#     """
#     Jacobian matrix of objective function wrt design variables
#     :param x_shared: vector of shared design variables
#     :type x_shared: ndarray
#     :param Y: vector of coupling variables
#     :type Y: ndarray
#     :returns: Jacobian matrix
#     """
#     return array([
#         2. - x_shared[2] * sqrt(10. * Y[6] / x_shared[0]) / Y[6] +
#         sqrt(x_shared[1]) + Y[0] / (2. * sqrt(x_shared[0] * Y[0])),
#         x_shared[0] / (2. * sqrt(x_shared[1])),
#         -2. * sqrt(10. * x_shared[0] / Y[6]),
#         1. - 2. * sqrt(10 * Y[1] / Y[6])
#     ])
# #
#
# def jacobian_f_y(x_shared, Y):
#     """
#     Jacobian matrix of objective function wrt coupling variables
#     :param x_shared: vector of shared design variables
#     :type x_shared: ndarray
#     :param Y: vector of coupling variables
#     :type Y: ndarray
#     :returns: Jacobian matrix
#     """
#     return array(
#         [Y[1] / (2. * sqrt(Y[0] * Y[1])) + x_shared[0] /
#          (2. * sqrt(x_shared[0] * Y[0])) + 1., -
#          x_shared[3] * sqrt(10. * Y[6] / Y[1]) / Y[6] - 2. *
#          Y[4] * sqrt(10. / Y[6]) +
#          Y[0] / (2. * sqrt(Y[0] * Y[1])) + 1., 0., 1., 1. - 2. * Y[1] *
#          sqrt(10. / Y[6]),
#          2.,
#          (Y[1] * x_shared[3] * sqrt(10. * Y[6] / Y[1]) + x_shared[0] *
# x_shared[2] *
#           sqrt(10. * Y[6] / x_shared[0])) / Y[6] ** 2 + Y[1] *
#          Y[4] * sqrt(10. / Y[6] ** 3)])
#
#
# def jacobian_residual_x(x_shared, Y):
#     """
#     Jacobian matrix of residual vector wrt design variables
#     :param x_shared: vector of shared design variables
#     :type x_shared: ndarray
#     :param Y: vector of coupling variables
#     :type Y: ndarray
#     :returns: Jacobian matrix
#     """
#     return array([
#         [1., 0., 0., 0.],
#         [-Y[2], 0., 0., 0.],
#         [0.1, 0., 0., 0.],
#         [0.2 * x_shared[0], 0., 0., 0.],
#         [0., 0., 1., 1.],
#         [0., 2., 0., 0.],
#         [-1., -1., -1., -1.]
#     ])
#
#
# def jacobian_residual_y(x_shared, Y):
#     """
#     Jacobian matrix of residual vector wrt coupling variables
#     :param x_shared: vector of shared design variables
#     :type x_shared: ndarray
#     :param Y: vector of coupling variables
#     :type Y: ndarray
#     :returns: Jacobian matrix
#     """
#     return array([
#         [0, 1., 0., 0., 0., 0., 0.],
#         [Y[1], Y[0], -Z[0], 0, 0, 0, 0],
#         [0., -40 * Y[3] / Y[6], 0., -40 * Y[1] / Y[6], 0., 0., 40 *
#          Y[1] * Y[3] / Y[6] ** 2],
#         [0., - 80 * Y[1] * Y[5] / Y[6], 0., 0., 0., -40 * Y[1] ** 2 / Y[6],
#          40 * Y[1] ** 2 * Y[5] / Y[6] ** 2],
#         [2., 0., 2., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 1., 0., 0.],
#         [-1., -1., -1., -1., -1., -1., 1.]])
