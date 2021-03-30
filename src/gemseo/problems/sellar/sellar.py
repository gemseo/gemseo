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
#        :author: Charlie Vanaret
#                 Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
# from Tedford2010
# Sellar analytical problem, p9
r"""
Sellar MDO problem
******************

The **sellar** module implements all :class:`.MDODiscipline`
included in the Sellar problem:

.. math::

   \begin{aligned}
   \text{minimize the objective function }&obj=x_{local}^2 + x_{shared,1}
   +y_0+e^{-y_1} \\
   \text{with respect to the design variables }&x_{shared},\,x_{local} \\
   \text{subject to the general constraints }
   & c_1 \leq 0\\
   & c_2 \leq 0\\
   \text{subject to the bound constraints }
   & -10 \leq x_{shared,0} \leq 10\\
   & 0 \leq x_{shared,1} \leq 10\\
   & 0 \leq x_{local} \leq 10.
   \end{aligned}

where the coupling variables are

.. math::

    \text{Discipline 0: } y_0 = x_{shared,0}^2 + x_{shared,1} +
     x_{local} - 0.2\,y_1,

and

.. math::

    \text{Discipline 1: }y_1 = \sqrt{y_0} + x_{shared,0} + x_{shared,1}.

and where the general constraints are

.. math::

   c_0 = 1-y_0/{3.16}

   c_1 = y_1/{24.} - 1
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import super
from cmath import exp, sqrt

from future import standard_library
from numpy import array, atleast_2d, complex128, ones, zeros

from gemseo.core.discipline import MDODiscipline

standard_library.install_aliases()


def get_inputs(names=None):
    """Generate initial solution

    :param names: input names (default: None)
    :type names: list(str)
    :returns: input values
    :rtype: dict
    """
    inputs = {
        "x_local": array([0.0], dtype=complex128),
        "x_shared": array([1.0, 0.0], dtype=complex128),
        "y_0": ones((1), dtype=complex128),
        "y_1": ones((1), dtype=complex128),
    }
    if names is None:
        return inputs
    return {k: inputs[k] for k in names}


class SellarSystem(MDODiscipline):
    """**SellarSystem** is the :class:`.MDODiscipline`
    implementing the computation of the Sellar's objective
    and constraints discipline."""

    def __init__(self):
        """Constructor"""
        super(SellarSystem, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = get_inputs()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def _run(self):
        """Defines the execution of the process, given that data
        has been checked.
        Compute the outputs (= objective value and constraints at system level)
        of the Sellar analytical problem.
        """
        x_local, x_shared, y_0, y_1 = self.get_inputs_by_name(
            ["x_local", "x_shared", "y_0", "y_1"]
        )
        obj = array([self.obj(x_local, x_shared, y_0, y_1)], dtype=complex128)
        c_1 = array([self.c_1(y_0)], dtype=complex128)
        c_2 = array([self.c_2(y_1)], dtype=complex128)
        self.store_local_data(obj=obj, c_1=c_1, c_2=c_2)

    @staticmethod
    def obj(x_local, x_shared, y_0, y_1):
        """Objective function

        :param x_local: local design variables
        :type x_local: ndarray
        :param x_shared: shared design variables
        :type x_shared: ndarray
        :param y_0: coupling variable from discipline 1
        :type y_0: ndarray
        :param y_1: coupling variable from discipline 2
        :type y_1: ndarray
        :returns: Objective value
        :rtype: float
        """
        return x_local[0] ** 2 + x_shared[1] + y_0[0] ** 2 + exp(-y_1[0])

    @staticmethod
    def c_1(y_0):
        """First constraint on system level

        :param y_0: coupling variable from discipline 1
        :type y_0: ndarray
        :returns: Value of the constraint
        :rtype: float
        """
        return 3.16 - y_0[0] ** 2

    @staticmethod
    def c_2(y_1):
        """Second constraint on system level

        :param y_1: coupling variable from discipline 2
        :type y_1: ndarray
        :returns: Value of the constraint
        :rtype: float
        """
        return y_1[0] - 24.0

    def _compute_jacobian(self, inputs=None, outputs=None):
        """
        Computes the jacobian

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        # Initialize all matrices to zeros
        self._init_jacobian(inputs, outputs, with_zeros=True)
        x_local, _, y_0, y_1 = self.get_inputs_by_name(
            ["x_local", "x_shared", "y_0", "y_1"]
        )
        self.jac["c_1"]["y_0"] = atleast_2d(array([-2.0 * y_0]))
        self.jac["c_2"]["y_1"] = ones((1, 1))
        self.jac["obj"]["x_local"] = atleast_2d(array([2.0 * x_local[0]]))
        self.jac["obj"]["x_shared"] = atleast_2d(array([0.0, 1.0]))
        self.jac["obj"]["y_0"] = atleast_2d(array([2.0 * y_0[0]]))
        self.jac["obj"]["y_1"] = atleast_2d(array([-exp(-y_1[0])]))


class Sellar1(MDODiscipline):
    """**Sellar1** is the :class:`.MDODiscipline`
    implementing the 1st set of equations: y_0.
    """

    def __init__(self, residual_form=False):
        """
        Constructor

        :param residual_form: if True only residuals are computed, no Ys
        :type residual_form: bool
        """
        self.residual_form = residual_form
        super(Sellar1, self).__init__(auto_detect_grammar_files=True)
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY
        if residual_form:
            self.residual_variables = ["r_0"]
            self.output_grammar.remove_item("y_0")

        else:
            self.output_grammar.remove_item("r_0")
            self.input_grammar.remove_item("y_0")

        self.default_inputs = get_inputs(self.input_grammar.get_data_names())

    def get_attributes_to_serialize(self):
        """Defines the attributes to be serialized
        Can be overloaded by disciplines

        :returns: the list of attributes names
        :rtype: list(str)
        """
        base_d = super(Sellar1, self).get_attributes_to_serialize()
        base_d.append("residual_form")
        return base_d

    def _run(self):
        """Defines the execution of the process, given that
        data has been checked.
        Solve a coupling equation in functional form and
        compute coupling variable y_0.
        """
        x_local, x_shared, y_1 = self.get_inputs_by_name(["x_local", "x_shared", "y_1"])

        if self.residual_form:
            y_0 = self.get_inputs_by_name("y_0")

            # residual form
            r_0_out = array(
                [self.compute_r_0(x_local, x_shared, y_0, y_1)], dtype=complex128
            )
            # store outputs
            self.store_local_data(r_0=r_0_out)
        else:
            # functional form
            y_0_out = array(
                [self.compute_y_0(x_local, x_shared, y_1)], dtype=complex128
            )
            self.store_local_data(y_0=y_0_out)

    @staticmethod
    def compute_y_0(x_local, x_shared, y_1):
        """Solve the first coupling equation in functional form.

        :param x_local: vector of design variables local to discipline 1
        :type x_local: ndarray
        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :param y_1: coupling variable of discipline 2
        :type y_1: ndarray
        :returns: coupling variable y_0 of discipline 1
        :rtype: float
        """
        return sqrt(x_shared[0] ** 2 + x_shared[1] + x_local[0] - 0.2 * y_1[0])

    @staticmethod
    def compute_r_0(x_local, x_shared, y_0, y_1):
        """Evaluate the first coupling equation in residual form.

        :param x_local: vector of design variables local to discipline 1
        :type x_local: ndarray
        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :param y_0: coupling variable of discipline 1
        :type y_0: ndarray
        :param y_1: coupling variable of discipline 2
        :type y_1: ndarray
        :returns: coupling variable y_0
        :rtype: float
        """
        return Sellar1.compute_y_0(x_local, x_shared, y_1) - y_0[0]

    def _compute_jacobian(self, inputs=None, outputs=None):
        """

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        # Initialize all matrices to zeros
        self._init_jacobian(inputs, outputs, with_zeros=True)
        x_local, x_shared, y_1 = self.get_inputs_by_name(["x_local", "x_shared", "y_1"])

        inv_denom = 1.0 / (self.compute_y_0(x_local, x_shared, y_1))
        self.jac["y_0"] = {}
        self.jac["y_0"]["x_local"] = atleast_2d(array([0.5 * inv_denom]))
        self.jac["y_0"]["x_shared"] = atleast_2d(
            array([x_shared[0] * inv_denom, 0.5 * inv_denom])
        )
        self.jac["y_0"]["y_1"] = atleast_2d(array([-0.1 * inv_denom]))

        if self.residual_form:
            self.jac["r_0"]["x_local"] = atleast_2d(self.jac["y_0"]["x_local"])
            self.jac["r_0"]["x_shared"] = atleast_2d(self.jac["y_0"]["x_shared"])
            self.jac["r_0"]["y_1"] = atleast_2d(self.jac["y_0"]["y_1"])
            self.jac["r_0"]["y_0"] = -ones((1, 1))
            del self.jac["y_0"]


class Sellar2(MDODiscipline):
    """**Sellar1** is the :class:`.MDODiscipline`
    implementing the 2nd set of equations: y_1
    """

    def __init__(self, residual_form=False):
        """
        Constructor

        :param residual_form: if True only residuals are computed, no Ys
        :type residual_form: bool
        """
        self.residual_form = residual_form
        super(Sellar2, self).__init__(auto_detect_grammar_files=True)
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY
        if residual_form:
            self.residual_variables = ["r_1"]
            self.output_grammar.remove_item("y_1")
        else:
            self.output_grammar.remove_item("r_1")
            self.input_grammar.remove_item("y_1")
        self.default_inputs = get_inputs(self.input_grammar.get_data_names())

    def get_attributes_to_serialize(self):
        """Defines the attributes to be serialized
        Can be overloaded by disciplines

        :returns: the list of attributes names
        :rtype: list(str)
        """
        base_d = super(Sellar2, self).get_attributes_to_serialize()
        base_d.append("residual_form")
        return base_d

    def _run(self):
        """Defines the execution of the process, given
        that data has been checked.
        Solve a coupling equation in functional form and compute coupling
        variable y1.
        """
        x_shared, y_0 = self.get_inputs_by_name(["x_shared", "y_0"])

        if self.residual_form:
            y_1 = self.get_inputs_by_name("y_1")
            # residual form
            r_1_out = array([self.compute_r_1(x_shared, y_0, y_1)], dtype=complex128)
            # store outputs
            self.store_local_data(r_1=r_1_out)
        else:
            # functional form
            y_1_out = array([self.compute_y1(x_shared, y_0)], dtype=complex128)
            self.store_local_data(y_1=y_1_out)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """
        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        # Initialize all matrices to zeros
        self._init_jacobian(inputs, outputs, with_zeros=True)
        y_0 = self.get_inputs_by_name("y_0")
        self.jac["y_1"] = {}
        self.jac["y_1"]["x_local"] = zeros((1, 1))
        self.jac["y_1"]["x_shared"] = ones((1, 2))
        if y_0[0] < 0.0:
            self.jac["y_1"]["y_0"] = -ones((1, 1))
        elif y_0[0] == 0.0:
            self.jac["y_1"]["y_0"] = zeros((1, 1))
        else:
            self.jac["y_1"]["y_0"] = ones((1, 1))

        if self.residual_form:
            self.jac["r_1"]["x_local"] = self.jac["y_1"]["x_local"]
            self.jac["r_1"]["x_shared"] = self.jac["y_1"]["x_shared"]
            self.jac["r_1"]["y_0"] = self.jac["y_1"]["y_0"]
            self.jac["r_1"]["y_1"] = -ones((1, 1))
            del self.jac["y_1"]

    @staticmethod
    def compute_y1(x_shared, y_0):
        """Solve the second coupling equation in functional form.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :param y_0: coupling variable of discipline 1
        :type y_0: ndarray
        :returns: coupling variable y_1
        :rtype: float
        """
        out = x_shared[0] + x_shared[1]
        if y_0[0].real == 0:
            y_1 = out
        elif y_0[0].real > 0:
            y_1 = y_0[0] + out
        else:
            y_1 = -y_0[0] + out
        return y_1

    @staticmethod
    def compute_r_1(x_shared, y_0, y_1):
        """Evaluate the second coupling equation in residual form.

        :param x_shared: vector of shared design variables
        :type x_shared: ndarray
        :param y_0: coupling variable of discipline 1
        :type y_0: ndarray
        :param y_1: coupling variable of discipline 2
        :type y_1: ndarray
        :returns: coupling variable y_0
        :rtype: float
        """
        return Sellar2.compute_y1(x_shared, y_0) - y_1[0]


# #=========================================================================
# # Analytical Jacobians of f and residuals
#
#
# class Jacobian(object):
#     """
#     Compute the Jacobians of objective function and residuals with respect
#     to design and coupling variables
#     """
#
#     @staticmethod
#     def jacobian_f_x(x_shared):
#         """
#         Jacobian matrix of objective function wrt design variables
#         :param x_shared: vector of design variables
#         :type x_shared: ndarray
#         :returns: Jacobian matrix
#         """
#         return array([0, 2 * x_shared[1], 1])
#
#     @staticmethod
#     def jacobian_f_y(Y):
#         """
#         Jacobian matrix of objective function wrt coupling variables
#         :param Y: vector of coupling variables
#         :type Y: ndarray
#         :returns: Jacobian matrix
#         """
#         return array([2 * Y[0], -exp(-Y[1])])
#
#     @staticmethod
#     def jacobian_residuals_x(x_shared):
#         """
#         Jacobian matrix of residual vector wrt design variables
#         :param x_shared: vector of design variables
#         :type x_shared: ndarray
#         :returns: Jacobian matrix
#         """
#         return array([[2 * x_shared[0], 1, 1], [1, 0, 1]])
#
#     @staticmethod
#     def jacobian_residuals_y(Y):
#         """
#         Jacobian matrix of residual vector wrt coupling variables
#         :param Y: vector of coupling variables
#         :type Y: ndarray
#         :returns: Jacobian matrix
#         """
#         return array([[-2 * Y[0], -0.2], [Y[0] / abs(Y[0]), -1.]])
#
#     @staticmethod
#     def jacobian_g_x():
#         """
#         Jacobian matrix of system-level constraints wrt design variables
#         :returns: Jacobian matrix
#         """
#         return array([[0., 0., 0.], [0., 0., 0.]])
#
#     @staticmethod
#     def jacobian_g_y(Y):
#         """
#         Jacobian matrix of system-level constraints wrt coupling variables
#         :param Y: vector of coupling variables
#         :type Y: ndarray
#         :returns: Jacobian matrix
#         """
#         return array([[2 * Y[0], 0], [0, 1]])
