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

"""
Functional operations
*********************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library
from numpy import delete, insert

from gemseo.core.function import MDOFunction

standard_library.install_aliases()


class RestrictedFunction(MDOFunction):
    """
    Restrict an MDOFunction to a subset of its input vector
    Fixes the rest of the indices
    """

    def __init__(self, orig_function, restriction_indices, restriction_values):
        """
        Constructor

        :param orig_function: the original function to restrict
        :param restriction_indices: indices array of the input vector to fix
        :param restriction_values: the values of the input vector at indices
            'restriction_indices' are set to restriction_values
        """
        if not restriction_indices.shape == restriction_values.shape:
            raise ValueError(
                "Inconsistent shapes for restriction" + "values and indices "
            )
        self.restriction_values = restriction_values
        func, jac = self._restrict(orig_function, restriction_indices)
        self._orig_function = orig_function
        name = str(orig_function.name) + "_restr"
        super(RestrictedFunction, self).__init__(
            func,
            name,
            jac=jac,
            f_type=orig_function.f_type,
            expr=orig_function.expr,
            args=orig_function.args,
            dim=orig_function.dim,
            outvars=orig_function.outvars,
        )

    def _restrict(self, orig_function, restriction_indices):
        """
        Restricts the function, builds the pointer to f and jac
        @param orig_function: original MDOFunction pointer
        @param restriction_indices: indices array for restriction
        """

        def restricted_function(x_vect):
            """Wrapped provided function in order to give to
            optimizer

            :param x_vect: design variable
            :returns: evaluation of function at x_vect
            """
            x_full = insert(x_vect, restriction_indices, self.restriction_values)
            return orig_function(x_full)

        def restricted_jac(x_vect):
            """Wrapped provided jacobian in order to give to
            optimizer

            :param x_vect: design variable
            :returns: evaluation of jacobian at x_vect
            """
            x_full = insert(x_vect, restriction_indices, self.restriction_values)
            jac = orig_function.jac(x_full)
            jac = delete(jac, restriction_indices, axis=0)
            return jac

        return restricted_function, restricted_jac


class LinerarComposition(MDOFunction):
    """
    Composes a function with a linear operator defined by a matrix
    computes orig_f(Mat.dot(x))
    """

    def __init__(self, orig_function, interp_operator):
        """
        Constructor

        :param orig_function: the original function to restrict
        :param interp_operator: operator matrix, the output of the
            function will be f(interp_operator.dot(x))
        """
        func, jac = self._restrict(orig_function, interp_operator)
        self._orig_function = orig_function
        super(LinerarComposition, self).__init__(
            func,
            str(orig_function.name) + "_comp",
            jac=jac,
            f_type=orig_function.f_type,
            expr="Mat*" + str(orig_function.expr),
            args=orig_function.args,
            dim=orig_function.dim,
            outvars=orig_function.outvars,
        )

    @staticmethod
    def _restrict(orig_function, interp_operator):
        """
        Generates the function restriction
        @param orig_function : the original function to restrict
        @param interp_operator: operator matrix, the output of the
            function will be f(interp_operator.dot(x))
        """

        def restricted_function(x_vect):
            """Wrapped provided function in order to give to
            optimizer

            :param x_vect: design variable
            :returns: evaluation of function at x_vect
            """
            x_full = interp_operator.dot(x_vect)
            return orig_function(x_full)

        def restricted_jac(x_vect):
            """Wrapped provided jacobian in order to give to
            optimizer

            :param x_vect: design variable
            :returns: evaluation of jacobian at x_vect
            """
            x_full = interp_operator.dot(x_vect)
            jac = orig_function.jac(x_full)
            return interp_operator.T.dot(jac)

        return restricted_function, restricted_jac
