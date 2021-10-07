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
#                        documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class to describe a function."""
from __future__ import division, unicode_literals

import logging
from numbers import Number
from typing import TYPE_CHECKING, Callable, Mapping, Sequence, Union

from numpy import hstack, ndarray, reshape, vstack

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.data_conversion import DataConversion

if TYPE_CHECKING:
    from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator


LOGGER = logging.getLogger(__name__)

OperandType = Union[ndarray, Number]
OperatorType = Callable[[OperandType, OperandType], OperandType]


class MakeFunction(MDOFunction):
    """A function object from io and reference data."""

    def __init__(
        self,
        input_names,  # type: Sequence[str]
        output_names,  # type: Sequence[str]
        default_inputs,  # type: Mapping[str, ndarray]
        mdo_function,  # type: MDOFunctionGenerator
    ):  # type: (...) -> None
        """
        Args:
            input_names: The dict keys of the input data.
            output_names: The dict keys of the output data.
            default_inputs: The default inputs dict to eventually overload
                the discipline's default inputs when evaluating the discipline.
            mdo_function: The MDOFunctionGenerator object.
        """
        self.__input_names = input_names
        self.__output_names = output_names
        self.__default_inputs = default_inputs
        self.__mdo_function = mdo_function

        default_name = "_".join(self.__output_names)

        super(MakeFunction, self).__init__(
            self._func,
            jac=self._func_jac,
            name=default_name,
            args=self.__input_names,
            outvars=self.__output_names,
        )

    def _func(
        self, x_vect  # type: ndarray
    ):  # type: (...) -> ndarray
        """A function which executes a discipline.

        Args:
            x_vect: The input vector of the function.

        Returns:
            The selected outputs of the discipline.
        """
        for name in self.__input_names:
            if name not in self.__mdo_function.discipline.default_inputs:
                raise ValueError(
                    "Discipline  {}"
                    " has no default_input named {}"
                    ", while input is required"
                    " by MDOFunction.".format(self.__mdo_function.discipline.name, name)
                )
        defaults = self.__mdo_function.discipline.default_inputs
        if self.__default_inputs is not None:
            defaults.update(self.__default_inputs)
        data = DataConversion.update_dict_from_array(
            defaults, self.__input_names, x_vect
        )
        self.__mdo_function.discipline.reset_statuses_for_run()
        computed_values = self.__mdo_function.discipline.execute(data)
        values_array = DataConversion.dict_to_array(
            computed_values, self.__output_names
        )
        if values_array.size == 1:  # Then the function is scalar
            return values_array[0]
        return values_array

    def _func_jac(
        self, x_vect  # type: ndarray
    ):  # type: (...) -> ndarray
        """A function which linearizes a discipline.

        Args:
            x_vect: The input vector of the function.

        Returns:
            The selected outputs of the discipline.
        """
        defaults = self.__mdo_function.discipline.default_inputs
        n_dv = len(x_vect)
        data = DataConversion.update_dict_from_array(
            defaults, self.__input_names, x_vect
        )
        self.__mdo_function.discipline.linearize(data)

        grad_array = []
        for out_name in self.__output_names:
            jac_loc = self.__mdo_function.discipline.jac[out_name]
            grad_loc = DataConversion.dict_to_array(jac_loc, self.__input_names)
            grad_output = hstack(grad_loc)
            if len(grad_output) > n_dv:
                grad_output = reshape(grad_output, (grad_output.size // n_dv, n_dv))
            grad_array.append(grad_output)
        grad = vstack(grad_array).real
        if grad.shape[0] == 1:
            grad = grad.flatten()
            assert len(x_vect) == len(grad)
        return grad
