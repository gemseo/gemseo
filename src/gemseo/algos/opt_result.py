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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import division, unicode_literals

import logging

from gemseo.utils.string_tools import MultiLineString

"""
Optimization result
*******************
"""


LOGGER = logging.getLogger(__name__)


class OptimizationResult(object):

    """Stores optimization results."""

    DICT_REPR_ATTR = [
        "x_0",
        "x_opt",
        "optimizer_name",
        "message",
        "f_opt",
        "status",
        "n_obj_call",
        "n_grad_call",
        "n_constr_call",
        "is_feasible",
    ]

    HDF_CSTR_GRAD_KEY = "constr_grad:"
    HDF_CSTR_KEY = "constr:"

    def __init__(
        self,
        x_0=None,
        x_opt=None,
        f_opt=None,
        status=None,
        constraints_values=None,
        constraints_grad=None,
        optimizer_name=None,
        message=None,
        n_obj_call=None,
        n_grad_call=None,
        n_constr_call=None,
        is_feasible=False,
    ):
        """Initialize optimization results.

        :param x_0: initial guess for design variables
        :param x_opt: optimal design variables values
        :param f_opt: the objective function values at optimum
        :param status: the optimizer status
        :param message: the optimizer message
        :param n_obj_call: number of call to objective function by optimizer
        :param n_grad_call: number of call to gradient function by optimizer
        :param n_constr_call: number of call to constraints
                function by optimizer
        :param is_feasible: True if the solution is feasible, false else
        """
        self.x_0 = x_0
        self.optimizer_name = optimizer_name
        self.x_opt = x_opt
        self.message = message
        self.f_opt = f_opt
        self.constraints_values = constraints_values
        self.constraints_grad = constraints_grad
        self.status = status
        self.n_obj_call = n_obj_call
        self.n_grad_call = n_grad_call
        self.n_constr_call = n_constr_call
        self.is_feasible = is_feasible

    def __repr__(self):
        msg = MultiLineString()
        msg.add("Optimization result:")
        msg.indent()
        msg.add("Design variables: {}", self.x_opt)
        msg.add("Objective function: {}", self.f_opt)
        msg.add("Feasible solution: {}", self.is_feasible)
        return str(msg)

    def __str__(self):
        msg = MultiLineString()
        msg.add("Optimization result:")
        msg.add("Objective value = {}", self.f_opt)
        if self.is_feasible:
            msg.add("The result is feasible.")
        else:
            msg.add("The result is not feasible.")
        msg.add("Status: {}", self.status)
        msg.add("Optimizer message: {}", self.message)
        if self.n_obj_call is not None:
            msg.add(
                "Number of calls to the objective function by the optimizer: {}",
                self.n_obj_call,
            )
        if self.constraints_values and len(self.constraints_values) < 20:
            msg.add("Constraints values:")
            msg.indent()
            for c_name in sorted(self.constraints_values.keys()):
                msg.add("{} = {}", c_name, self.constraints_values[c_name])
        return str(msg)

    def get_data_dict_repr(self):
        """Returns a dict representation of self for serialization functions are
        removed.

        :returns: a dict with attributes names as keys
        """
        repr_dict = {}
        for attr_name in self.DICT_REPR_ATTR:
            attr = getattr(self, attr_name)
            if attr is not None:
                repr_dict[attr_name] = attr

        cgrad_key = OptimizationResult.HDF_CSTR_GRAD_KEY
        c_key = OptimizationResult.HDF_CSTR_KEY
        for cstr, cval in self.constraints_values.items():
            repr_dict[c_key + cstr] = cval
        for cstr, cgrad in self.constraints_grad.items():
            repr_dict[cgrad_key + cstr] = cgrad

        return repr_dict

    @staticmethod
    def init_from_dict_repr(**kwargs):
        """Initializes a new opt result from a data dict typically used for
        deserialization.

        :param kwargs: key value pairs from DICT_REPR_ATTR
        """
        allowed = OptimizationResult.DICT_REPR_ATTR
        filt_args = {k: v for k, v in kwargs.items() if k in allowed}
        non_allowed = {k: v for k, v in kwargs.items() if k not in filt_args}
        constraints_values = {}
        constraints_grad = {}
        cgrad_key = OptimizationResult.HDF_CSTR_GRAD_KEY
        c_key = OptimizationResult.HDF_CSTR_KEY
        for attr, val in non_allowed.items():
            if attr.startswith(cgrad_key):
                key_clean = attr.replace(cgrad_key, "")
                constraints_grad[key_clean] = val
            elif attr.startswith(c_key):
                key_clean = attr.replace(c_key, "")
                constraints_values[key_clean] = val
            else:
                raise ValueError("Unknown attribute : " + str(attr))
        opt_res = OptimizationResult(
            constraints_values=constraints_values,
            constraints_grad=constraints_grad,
            **filt_args
        )
        return opt_res
