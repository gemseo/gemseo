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
"""A function computing some outputs of a discipline from some inputs."""
from __future__ import annotations

import logging
from numbers import Number
from typing import Callable
from typing import Iterable
from typing import Mapping
from typing import TYPE_CHECKING
from typing import Union

from numpy import empty
from numpy import ndarray

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import update_dict_of_arrays_from_array

if TYPE_CHECKING:
    from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator


LOGGER = logging.getLogger(__name__)

OperandType = Union[ndarray, Number]
OperatorType = Callable[[OperandType, OperandType], OperandType]


class MakeFunction(MDOFunction):
    """A function computing some outputs of a discipline from some inputs."""

    def __init__(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str],
        default_inputs: Mapping[str, ndarray] | None,
        mdo_function: MDOFunctionGenerator,
    ) -> None:
        """
        Args:
            input_names: The names of the inputs.
            output_names: The names of the outputs.
            default_inputs: The default values of the inputs
                to overload the default values of the inputs of the discipline.
                If None, do no overload them.
            mdo_function: The generator of the :class:`.MDOFunction`
                computing ``output_names`` from ``input_names``
                based on a :class:`.MDODiscipline`.
        """
        self.__input_names = input_names
        self.__output_names = output_names
        self.__mdo_function = mdo_function
        self.__default_inputs = default_inputs
        self.__input_indices = None
        self.__output_indices = None
        self.__output_size = 0
        self.__input_size = 0
        self.__jacobian = None
        self.__discipline = self.__mdo_function.discipline
        super().__init__(
            self._func,
            jac=self._func_jac,
            name="_".join(self.__output_names),
            args=self.__input_names,
            outvars=self.__output_names,
        )

    def __compute_input_indices(self):
        """Compute the indices of the input variables in the Jacobian array."""
        start = 0
        self.__input_size = 0
        self.__input_indices = {}
        for name in self.__input_names:
            jac = self.__discipline.jac[self.__output_names[0]][name]
            self.__input_size += jac.shape[1]
            self.__input_indices[name] = slice(start, self.__input_size)
            start = self.__input_size

    def __compute_output_indices(self):
        """Compute the indices of the input variables in the Jacobian array."""
        start = 0
        self.__output_size = 0
        self.__output_indices = {}
        for name in self.__output_names:
            jac = self.__discipline.jac[name][self.__input_names[0]]
            self.__output_size += jac.shape[0]
            self.__output_indices[name] = slice(start, self.__output_size)
            start = self.__output_size

    @property
    def _default_inputs(self) -> dict[str, ndarray]:
        """The default values of the inputs of the function at execution time.

        They correspond to the default values of the discipline when calling this
        property, and updated with :attr:`.__default_inputs` if not None.
        """
        default_inputs = self.__discipline.default_inputs
        if self.__default_inputs is not None:
            default_inputs.update(self.__default_inputs)

        return default_inputs

    def _func(self, x_vect: ndarray) -> OperandType:
        """A function which executes a discipline for specific inputs and outputs.

        Args:
            x_vect: The input data of the function.

        Returns:
            The output data of the function.
        """
        for input_name in self.__input_names:
            if input_name not in self.__discipline.default_inputs:
                raise ValueError(
                    "Discipline {} has no default input named {},"
                    "while input is required by MDOFunction.".format(
                        self.__discipline.name, input_name
                    )
                )

        self.__discipline.reset_statuses_for_run()
        input_data = self.__compute_input_data(x_vect)
        output_data = self.__discipline.execute(input_data)
        output_data = concatenate_dict_of_arrays_to_array(
            output_data, self.__output_names
        )
        if output_data.size == 1:  # Then the function is scalar
            return output_data[0]

        return output_data

    def _func_jac(self, x_vect: ndarray) -> ndarray:
        """A function which linearizes a discipline for specific inputs and outputs.

        Args:
            x_vect: The input data of the function.

        Returns:
            The Jacobian of the discipline for specific inputs and outputs.
        """
        self.__discipline.linearize(self.__compute_input_data(x_vect))
        if self.__jacobian is None:
            self.__compute_input_indices()
            self.__compute_output_indices()
            if self.__output_size == 1:
                self.__jacobian = empty(self.__input_size)
            else:
                self.__jacobian = empty((self.__output_size, self.__input_size))

        if self.__output_size == 1:
            output_name = self.__output_names[0]
            for input_name in self.__input_names:
                in_indices = self.__input_indices[input_name]
                jac = self.__discipline.jac[output_name][input_name]
                self.__jacobian[in_indices] = jac[0, :]
        else:
            for output_name in self.__output_names:
                out_indices = self.__output_indices[output_name]
                for input_name in self.__input_names:
                    in_indices = self.__input_indices[input_name]
                    jac = self.__discipline.jac[output_name][input_name]
                    self.__jacobian[out_indices, in_indices] = jac

        return self.__jacobian

    def __compute_input_data(
        self,
        x_vect: ndarray,
    ) -> dict[str, ndarray]:
        """Return the input data of the underlying discipline.

        Args:
            x_vect: The input vector of the function.

        Returns:
            The input data of the underlying discipline.
        """
        return update_dict_of_arrays_from_array(
            self._default_inputs, self.__input_names, x_vect, copy=False
        )
