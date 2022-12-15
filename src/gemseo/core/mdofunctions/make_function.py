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
"""A function computing some outputs of a discipline from some of its inputs."""
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

from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array

if TYPE_CHECKING:
    from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator


LOGGER = logging.getLogger(__name__)

OperandType = Union[ndarray, Number]
OperatorType = Callable[[OperandType, OperandType], OperandType]


class MakeFunction(MDOFunction):
    """A function executing and linearizing a discipline for some inputs and outputs."""

    def __init__(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str],
        default_inputs: Mapping[str, ndarray] | None,
        mdo_function: MDOFunctionGenerator,
        names_to_sizes: dict[str, int] | None = None,
    ) -> None:
        """
        Args:
            input_names: The names of the inputs.
            output_names: The names of the outputs.
            default_inputs: The default input values
                to overload the ones of the underlying discipline
                attached to the ``mdo_function``
                at each evaluation of the outputs with :meth:`._fun`
                or their derivatives with :meth:`._jac`.
                If ``None``, do not overload them.
            mdo_function: The generator of the :class:`.MDOFunction`
                based on a :class:`.MDODiscipline`.
            names_to_sizes: The sizes of the input variables.
                If ``None``, guess them from the default inputs and local data
                of the discipline :class:`.MDODiscipline`.
        """  # noqa: D205, D212, D415
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
        self.__names_to_indices = {}
        self.__names_to_sizes = names_to_sizes or {}
        super().__init__(
            self._func_to_wrap,
            jac=self._jac_to_wrap,
            name="_".join(self.__output_names),
            args=self.__input_names,
            outvars=self.__output_names,
        )

    def __compute_input_indices(self) -> None:
        """Compute the indices of the input variables in the Jacobian array."""
        start = 0
        self.__input_size = 0
        self.__input_indices = {}
        for name in self.__input_names:
            jac = self.__discipline.jac[self.__output_names[0]][name]
            self.__input_size += jac.shape[1]
            self.__input_indices[name] = slice(start, self.__input_size)
            start = self.__input_size

    def __compute_output_indices(self) -> None:
        """Compute the indices of the input variables in the Jacobian array."""
        start = 0
        self.__output_size = 0
        self.__output_indices = {}
        for name in self.__output_names:
            jac = self.__discipline.jac[name][self.__input_names[0]]
            self.__output_size += jac.shape[0]
            self.__output_indices[name] = slice(start, self.__output_size)
            start = self.__output_size

    def _func_to_wrap(self, x_vect: ArrayType) -> OperandType:
        """Compute an output vector from an input one.

        Args:
            x_vect: The input vector.

        Returns:
            The output vector.
        """
        self.__discipline.reset_statuses_for_run()
        input_data = self.__compute_discipline_input_data(x_vect)
        output_data = self.__discipline.execute(input_data)
        output_data = concatenate_dict_of_arrays_to_array(
            output_data, self.__output_names
        )
        if output_data.size == 1:  # Then the function is scalar
            return output_data[0]

        return output_data

    def _jac_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Compute the Jacobian value from an input vector.

        Args:
            x_vect: The input vector.

        Returns:
            The Jacobian value.
        """
        self.__discipline.linearize(self.__compute_discipline_input_data(x_vect))
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

    def __create_names_to_indices(self) -> None:
        """Create the map from discipline input names to input vector indices.

        Raises:
            ValueError: When a discipline input has no default value.
        """
        if set(self.__names_to_sizes) != set(self.__input_names):
            self.__names_to_sizes.update(
                {
                    name: value.size
                    for name, value in self.__discipline.get_input_data().items()
                    if name in self.__input_names
                }
            )
            self.__names_to_sizes.update(
                {
                    name: value.size
                    for name, value in self.__discipline.default_inputs.items()
                    if name in self.__input_names
                }
            )

        for input_name in self.__input_names:
            if input_name not in self.__names_to_sizes:
                raise ValueError(
                    f"The size of the input {input_name} cannot be guessed "
                    f"from the discipline {self.__discipline.name}, "
                    f"nor from its default inputs or from its local data."
                )

        index = 0
        for name in self.__input_names:
            size = self.__names_to_sizes[name]
            self.__names_to_indices[name] = slice(index, index + size)
            index += size

    def __compute_discipline_input_data(
        self,
        x_vect: ndarray,
    ) -> dict[str, ndarray]:
        """Return the input data of the underlying discipline.

        Args:
            x_vect: The input vector of the function.

        Returns:
            The input data of the underlying discipline.

        Raises:
            ValueError: When a discipline input has no default value.
        """
        if self.__default_inputs is not None:
            self.__discipline.default_inputs.update(self.__default_inputs)

        if not self.__names_to_indices:
            self.__create_names_to_indices()

        return {
            name: x_vect[self.__names_to_indices[name]] for name in self.__input_names
        }
