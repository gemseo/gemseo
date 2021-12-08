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
"""Optimization result."""
from __future__ import division, unicode_literals

import logging
from typing import Dict, Mapping, Optional, Union

from numpy import ndarray

from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)


class OptimizationResult(object):
    """Store the result of an optimization."""

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
        "optimum_index",
    ]

    HDF_CSTR_GRAD_KEY = "constr_grad:"
    HDF_CSTR_KEY = "constr:"

    def __init__(
        self,
        x_0=None,  # type: Optional[ndarray]
        x_opt=None,  # type: Optional[ndarray]
        f_opt=None,  # type: Optional[ndarray]
        status=None,  # type: Optional[int]
        constraints_values=None,  # type: Optional[Mapping[str,ndarray]]
        constraints_grad=None,  # type: Optional[Mapping[str,ndarray]]
        optimizer_name=None,  # type: Optional[str]
        message=None,  # type: Optional[str]
        n_obj_call=None,  # type: Optional[int]
        n_grad_call=None,  # type: Optional[int]
        n_constr_call=None,  # type: Optional[int]
        is_feasible=False,  # type: bool
        optimum_index=None,  # type: Optional[int]
    ):  # type: (...) -> None
        # noqa: E262
        """
        Args:
            x_0: The initial values of the design variables.
            x_opt: The optimal values of the design variables, called the *optimum*.
            f_opt: The value of the objective function at the optimum.
            status: The status of the optimization.
            constraints_values: The values of the constraints at the optimum.
            constraints_grad: The values of the gradients of the constraints
                at the optimum.
            optimizer_name: The name of the optimizer.
            message: The message returned by the optimizer.
            n_obj_call: The number of calls to the objective function.
            n_grad_call: The number of calls to the gradient function.
            n_constr_call: The number of calls to the constraints function.
            is_feasible: Whether the solution is feasible.
            optimum_index: The position of the optimum in the optimization history,
                0 being the first one.
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
        self.optimum_index = optimum_index

    def __repr__(self):  # type: (...) -> str
        msg = MultiLineString()
        msg.add("Optimization result:")
        msg.indent()
        msg.add("Design variables: {}", self.x_opt)
        msg.add("Objective function: {}", self.f_opt)
        msg.add("Feasible solution: {}", self.is_feasible)
        return str(msg)

    def __str__(self):  # type: (...) -> str
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
            for name, value in sorted(self.constraints_values.items()):
                msg.add("{} = {}", name, value)

        return str(msg)

    def get_data_dict_repr(
        self,
    ):  # type: (...) -> Dict[str,Union[str,int,bool,ndarray]]
        """Return a dictionary representation for serialization.

        The functions are removed.

        Returns:
            A dictionary representation of the optimization result.
        """
        representation = {}
        for attribute_name in self.DICT_REPR_ATTR:
            attribute_value = getattr(self, attribute_name)
            if attribute_value is not None:
                representation[attribute_name] = attribute_value

        for name, value in self.constraints_values.items():
            representation[self.HDF_CSTR_KEY + name] = value

        for name, value in self.constraints_grad.items():
            representation[self.HDF_CSTR_GRAD_KEY + name] = value

        return representation

    @staticmethod
    def init_from_dict_repr(**kwargs):  # type: (...) -> OptimizationResult
        """Initialize a new optimization result from a data dictionary.

        This is typically used for deserialization.

        Args:
            **kwargs: The data whose names are :attr:`.DICT_REPR_ATTR`.

        Returns:
            The optimization result defined by the passed data.

        Raises:
            ValueError: When a data name is unknown.
        """
        allowed_kwargs = {
            k: v for k, v in kwargs.items() if k in OptimizationResult.DICT_REPR_ATTR
        }
        non_allowed_kwargs = {
            k: v for k, v in kwargs.items() if k not in allowed_kwargs
        }
        constraints_values = {}
        constraints_grad = {}
        cgrad_key = OptimizationResult.HDF_CSTR_GRAD_KEY
        c_key = OptimizationResult.HDF_CSTR_KEY
        for kwarg_name, kwarg_value in non_allowed_kwargs.items():
            if kwarg_name.startswith(cgrad_key):
                constraints_grad[kwarg_name.replace(cgrad_key, "")] = kwarg_value
            elif kwarg_name.startswith(c_key):
                constraints_values[kwarg_name.replace(c_key, "")] = kwarg_value
            else:
                raise ValueError("Unknown attribute: {}.".format(kwarg_name))

        return OptimizationResult(
            constraints_values=constraints_values,
            constraints_grad=constraints_grad,
            **allowed_kwargs
        )
