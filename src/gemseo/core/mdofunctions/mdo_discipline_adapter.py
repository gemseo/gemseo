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

from numbers import Number
from typing import TYPE_CHECKING
from typing import Callable
from typing import Union

from numpy import array
from numpy import empty
from numpy import ndarray

from gemseo.core.mdofunctions.linear_candidate_function import LinearCandidateFunction
from gemseo.utils.compatibility.scipy import sparse_classes

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from collections.abc import Sequence

    from gemseo import MDODiscipline
    from gemseo.core.mdofunctions.mdo_function import ArrayType

OperandType = Union[ndarray, Number]
OperatorType = Callable[[OperandType, OperandType], OperandType]


class MDODisciplineAdapter(LinearCandidateFunction):
    """An :class:`.MDOFunction` executing a discipline for some inputs and outputs."""

    __linear_candidate: bool
    """Whether the final MDOFunction could be linear."""

    __input_dimension: int | None
    """The input variable dimension, needed for linear candidates."""

    def __init__(
        self,
        input_names: Sequence[str],
        output_names: Sequence[str],
        default_inputs: Mapping[str, ndarray] | None,
        discipline: MDODiscipline,
        names_to_sizes: MutableMapping[str, int] | None = None,
        linear_candidate: bool = False,
    ) -> None:
        """
        Args:
            input_names: The names of the inputs.
            output_names: The names of the outputs.
            default_inputs: The default input values
                to overload the ones of the discipline
                at each evaluation of the outputs with :meth:`._fun`
                or their derivatives with :meth:`._jac`.
                If ``None``, do not overload them.
            discipline: The discipline to be adapted.
            names_to_sizes: The sizes of the input variables.
                If ``None``, determine them from the default inputs and local data
                of the discipline :class:`.MDODiscipline`.
            linear_candidate: Whether the final MDOFunction could be linear.
        """  # noqa: D205, D212, D415
        self.__input_names = input_names
        self.__output_names = output_names
        self.__default_inputs = default_inputs if default_inputs is not None else {}
        self.__input_size = 0
        self.__output_names_to_slices = {}
        self.__jacobian = array(())
        self.__discipline = discipline
        self.__input_names_to_slices = {}
        self.__input_names_to_sizes = (
            names_to_sizes if names_to_sizes is not None else {}
        )
        self.__linear_candidate = linear_candidate
        self.__input_dimension = self.__compute_input_dimension(
            default_inputs, discipline, input_names
        )
        super().__init__(
            self._func_to_wrap,
            jac=self._jac_to_wrap,
            name="_".join(self.__output_names),
            input_names=self.__input_names,
            output_names=self.__output_names,
        )

    @property
    def linear_candidate(self) -> bool:  # noqa: D102
        return self.__linear_candidate

    @property
    def input_dimension(self) -> int | None:  # noqa: D102
        return self.__input_dimension

    def __compute_input_dimension(
        self,
        default_inputs: Mapping[str, ndarray] | None,
        discipline: MDODiscipline,
        input_names: Sequence[str],
    ) -> int | None:
        """Compute the input dimension.

        Args:
            default_inputs: : The default input values
                to overload the ones of the discipline
                at each evaluation of the outputs with :meth:`._fun`
                or their derivatives with :meth:`._jac`.
                If ``None``, do not overload them.
            discipline: The discipline to be adapted.
            input_names: The names of the inputs.

        Returns:
            The input dimension.
        """
        if default_inputs and all(inpt in default_inputs for inpt in input_names):
            return sum([
                len(default_inputs[inpt])
                if isinstance(default_inputs[inpt], ndarray)
                else 1
                for inpt in input_names
            ])

        if len(self.__input_names_to_sizes) > 0:
            return sum(self.__input_names_to_sizes.values())

        if all(inpt in discipline.default_inputs for inpt in input_names):
            return sum([
                len(discipline.default_inputs[inpt])
                if isinstance(discipline.default_inputs[inpt], ndarray)
                else 1
                for inpt in input_names
            ])

        return None

    def __create_output_names_to_slices(self) -> int:
        """Compute the indices of the input variables in the Jacobian array.

        Returns:
            The size of the inputs.
        """
        self.__output_names_to_slices = output_names_to_slices = {}
        start = 0
        output_size = 0
        jac_row_id = self.__input_names[0]
        jac = self.__discipline.jac
        for name in self.__output_names:
            output_size += jac[name][jac_row_id].shape[0]
            output_names_to_slices[name] = slice(start, output_size)
            start = output_size
        return output_size

    def _func_to_wrap(self, x_vect: ArrayType) -> ndarray | Number:
        """Compute an output vector from an input one.

        Args:
            x_vect: The input vector.

        Returns:
            The output vector or a scalar if the vector has only one component.
        """
        self.__discipline.reset_statuses_for_run()
        input_data = self.__create_discipline_input_data(x_vect)
        output_data = self.__discipline.execute(input_data)
        output_array = (
            self.__discipline.output_grammar.data_converter.convert_data_to_array(
                self.__output_names,
                output_data,
            )
        )

        if output_array.size == 1:
            # The function is scalar.
            return output_array[0]

        return output_array

    def _jac_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Compute the Jacobian value from an input vector.

        Args:
            x_vect: The input vector.

        Returns:
            The Jacobian value.
        """
        self.__discipline.linearize(self.__create_discipline_input_data(x_vect))

        if len(self.__jacobian) == 0:
            output_size = self.__create_output_names_to_slices()
            if output_size == 1:
                shape = self.__input_size
            else:
                shape = (output_size, self.__input_size)
            self.__jacobian = empty(shape)

        if self.__jacobian.ndim == 1 or self.__jacobian.shape[0] == 1:
            output_name = self.__output_names[0]
            jac_output = self.__discipline.jac[output_name]
            for input_name in self.__input_names:
                input_slice = self.__input_names_to_slices[input_name]
                jac = jac_output[input_name]
                # TODO: This precaution is meant to disappear when sparse 1-D array will
                # be available. This is also mandatory since self.__jacobian is
                # initialized as a dense array.
                if isinstance(jac, sparse_classes):
                    first_row = jac.getrow(0).todense().flatten()
                else:
                    first_row = jac[0, :]

                self.__jacobian[input_slice] = first_row
        else:
            for output_name in self.__output_names:
                output_slice = self.__output_names_to_slices[output_name]
                jac_output = self.__discipline.jac[output_name]
                for input_name in self.__input_names:
                    input_slice = self.__input_names_to_slices[input_name]
                    jac = jac_output[input_name]
                    # TODO: This is mandatory since self.__jacobian is initialized as a
                    # dense array. Performance improvement could be obtained if one is
                    # able to infer the type of jac.
                    if isinstance(jac, sparse_classes):
                        jac = jac.toarray()

                    self.__jacobian[output_slice, input_slice] = jac

        return self.__jacobian

    def __create_input_names_to_slices(self) -> None:
        """Create the map from discipline input names to input vector slices.

        Raises:
            ValueError: When a discipline input has no default value.
        """
        input_data = self.__discipline.get_input_data()
        input_data.update(self.__discipline.default_inputs)

        missing_names = (
            set(self.__input_names)
            .difference(self.__input_names_to_sizes.keys())
            .difference(input_data.keys())
        )

        if missing_names:
            raise ValueError(
                f"The size of the input {','.join(missing_names)} cannot be guessed "
                f"from the discipline {self.__discipline.name}, "
                f"nor from its default inputs or from its local data."
            )

        (
            self.__input_names_to_slices,
            self.__input_size,
        ) = self.__discipline.input_grammar.data_converter.compute_names_to_slices(
            self.__input_names, input_data, self.__input_names_to_sizes
        )

    def __create_discipline_input_data(
        self,
        x_vect: ndarray,
    ) -> dict[str, ndarray]:
        """Return the input data for the underlying discipline.

        The variables in the input data are cast according to the types defined in the
        design space.

        Args:
            x_vect: The input vector of the function.

        Returns:
            The input data of the underlying discipline.

        Raises:
            ValueError: When a discipline input has no default value.
        """
        if self.__default_inputs:
            self.__discipline.default_inputs.update(self.__default_inputs)

        if not self.__input_names_to_slices:
            self.__create_input_names_to_slices()

        input_data = (
            self.__discipline.input_grammar.data_converter.convert_array_to_data(
                x_vect, self.__input_names_to_slices
            )
        )

        variable_types = x_vect.dtype.metadata

        if variable_types is not None:
            # Restore the proper data types as declared in the design space.
            for name, type_ in variable_types.items():
                input_data[name] = input_data[name].astype(type_, copy=False)

        return input_data
