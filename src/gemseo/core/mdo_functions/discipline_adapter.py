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

from typing import TYPE_CHECKING

from numpy import array
from numpy import empty
from numpy import ndarray

from gemseo.core.execution_status import ExecutionStatus
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.utils.compatibility.scipy import get_row
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.core.grammars.grammar_properties import GrammarProperties
    from gemseo.typing import JacobianData
    from gemseo.typing import NumberArray
    from gemseo.typing import StrKeyMapping


class DisciplineAdapter(MDOFunction):
    """An :class:`.MDOFunction` executing a discipline for some inputs and outputs."""

    __is_linear: bool
    """Whether the function is linear."""

    __input_dimension: int | None
    """The input variable dimension, needed for linear candidates."""

    differentiated_input_names_substitute: Sequence[str]
    """The names of the inputs with respect to which to differentiate the functions.

    If empty, consider the variables of their input space.
    """

    def __init__(
        self,
        input_names: Sequence[str],
        output_names: Sequence[str],
        default_input_data: GrammarProperties,
        discipline: Discipline,
        names_to_sizes: MutableMapping[str, int] = READ_ONLY_EMPTY_DICT,
        differentiated_input_names_substitute: Sequence[str] = (),
    ) -> None:
        """
        Args:
            input_names: The names of the inputs.
            output_names: The names of the outputs.
            default_input_data: The default input values
                to overload the ones of the discipline
                at each evaluation of the outputs with :meth:`._fun`
                or their derivatives with :meth:`._jac`.
                If empty, do not overload them.
            discipline: The discipline to be adapted.
            names_to_sizes: The sizes of the input variables.
                If empty, determine them from the default inputs and local data
                of the discipline :class:`.Discipline`.
            differentiated_input_names_substitute: The names of the inputs
                with respect to which to differentiate the functions.
                If empty, consider the variables of their input space.
        """  # noqa: D205, D212, D415
        super().__init__(
            self._func_to_wrap,
            jac=self._jac_to_wrap,
            name="_".join(output_names),
            input_names=input_names,
            output_names=output_names,
        )
        self.differentiated_input_names_substitute = (
            differentiated_input_names_substitute or input_names
        )
        self.__default_inputs = default_input_data
        self.__input_size = 0
        self.__differentiated_input_size = 0
        self.__output_names_to_slices = {}
        self.__jacobian = array(())
        self.__discipline = discipline
        self.__input_names_to_slices = {}
        self.__input_names_to_sizes = names_to_sizes or {}
        self.__differentiated_input_names_to_slices = {}
        input_names = set(self.input_names)
        self.__is_linear = self.__discipline.io.have_linear_relationships(
            input_names, output_names
        )
        self.__input_dimension = self.__compute_input_dimension(default_input_data)
        self.__convert_array_to_data = (
            discipline.io.input_grammar.data_converter.convert_array_to_data
        )

    @property
    def is_linear(self) -> bool:  # noqa: D102
        return self.__is_linear

    @property
    def input_dimension(self) -> int | None:  # noqa: D102
        return self.__input_dimension

    def __compute_input_dimension(
        self,
        default_input_data: GrammarProperties,
    ) -> int | None:
        """Compute the input dimension.

        Args:
            default_input_data: : The default input values
                to overload the ones of the discipline
                at each evaluation of the outputs with :meth:`._fun`
                or their derivatives with :meth:`._jac`.
                If ``None``, do not overload them.

        Returns:
            The input dimension.
        """
        get_value_size = (
            self.__discipline.io.input_grammar.data_converter.get_value_size
        )

        if default_input_data and all(
            name in default_input_data for name in self.input_names
        ):
            return sum(
                get_value_size(input_name, default_input_data[input_name])
                for input_name in self.input_names
            )

        if len(self.__input_names_to_sizes) > 0:
            return sum(self.__input_names_to_sizes.values())

        default_input_data = self.__discipline.io.input_grammar.defaults

        if all(name in default_input_data for name in self.input_names):
            return sum(
                get_value_size(input_name, default_input_data[input_name])
                for input_name in self.input_names
            )

        # TODO: document what None means. We could use 0 instead.
        return None

    def __create_output_names_to_slices(self, jacobians: JacobianData) -> int:
        """Compute the indices of the input variables in the Jacobian array.

        Args:
            jacobians: The Jacobians data used to compute the slices.

        Returns:
            The size of the inputs.
        """
        self.__output_names_to_slices = output_names_to_slices = {}
        start = 0
        output_size = 0
        for output_name in self.output_names:
            input_name = next(iter(jacobians[output_name]))
            output_size += jacobians[output_name][input_name].shape[0]
            output_names_to_slices[output_name] = slice(start, output_size)
            start = output_size
        return output_size

    def _func_to_wrap(self, x_vect: NumberArray) -> complex | NumberArray:
        """Compute an output vector from an input one.

        Args:
            x_vect: The input vector.

        Returns:
            The output vector or a scalar if the vector has only one component.
        """
        self.__discipline.execution_status.value = ExecutionStatus.Status.DONE
        input_data = self.__create_discipline_input_data(x_vect)
        output_data = self.__discipline.execute(input_data)
        return self._convert_output_data_to_array(output_data)

    def _convert_output_data_to_array(
        self, output_data: StrKeyMapping
    ) -> complex | NumberArray:
        """Convert the discipline's output data to array/scalar.

        Args:
            output_data: The discipline's output data.

        Returns:
            The vector or scalar of output data.
        """
        output_vector = (
            self.__discipline.io.output_grammar.data_converter.convert_data_to_array(
                self.output_names, output_data
            )
        )

        if output_vector.size == 1:  # The function is scalar.
            return output_vector[0]

        return output_vector

    def _jac_to_wrap(self, x_vect: NumberArray) -> NumberArray:
        """Compute the Jacobian value from an input vector.

        Args:
            x_vect: The input vector.

        Returns:
            The Jacobian value.
        """
        input_data = self.__create_discipline_input_data(x_vect)
        jacobians = self.__discipline.linearize(input_data)

        return self._convert_jacobian_to_array(jacobians)

    def _convert_jacobian_to_array(self, jacobians: JacobianData) -> NumberArray:
        """Convert the discipline's Jacobians to array.

        Args:
            jacobians: The discipline's Jacobians data.

        Returns:
            The aggregated Jacobian as a NumPy array.
        """
        if len(self.__jacobian) == 0:
            output_size = self.__create_output_names_to_slices(jacobians)
            if output_size == 1:
                shape = self.__differentiated_input_size
            else:
                shape = (output_size, self.__differentiated_input_size)

            self.__jacobian = empty(shape)

        if self.__jacobian.ndim == 1 or self.__jacobian.shape[0] == 1:
            output_name = self.output_names[0]
            jac_output = jacobians[output_name]
            for input_name in self.differentiated_input_names_substitute:
                input_slice = self.__differentiated_input_names_to_slices[input_name]
                jac = jac_output[input_name]
                # TODO: This precaution is meant to disappear when sparse 1-D array will
                # be available. This is also mandatory since self.__jacobian is
                # initialized as a dense array.
                if isinstance(jac, sparse_classes):
                    first_row = get_row(jac, 0).todense().flatten()
                else:
                    first_row = jac[0, :]

                self.__jacobian[input_slice] = first_row
        else:
            for output_name in self.output_names:
                output_slice = self.__output_names_to_slices[output_name]
                jac_output = jacobians[output_name]
                for input_name in self.differentiated_input_names_substitute:
                    input_slice = self.__differentiated_input_names_to_slices[
                        input_name
                    ]
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
        input_data = self.__discipline.io.get_input_data()
        input_data.update(self.__discipline.io.input_grammar.defaults)
        self.__input_names_to_sizes.update(
            self.__discipline.io.input_grammar.data_converter.compute_names_to_sizes(
                input_data.keys(), input_data
            )
        )

        missing_names = (
            set(self.input_names)
            .difference(self.__input_names_to_sizes.keys())
            .difference(input_data.keys())
        )

        if missing_names:
            msg = (
                f"The size of the input {','.join(missing_names)} cannot be guessed "
                f"from the discipline {self.__discipline.name}, "
                f"nor from its default inputs or from its local data."
            )
            raise ValueError(msg)

        (
            self.__input_names_to_slices,
            self.__input_size,
        ) = self.__discipline.io.input_grammar.data_converter.compute_names_to_slices(
            self.input_names, input_data, self.__input_names_to_sizes
        )
        (
            self.__differentiated_input_names_to_slices,
            self.__differentiated_input_size,
        ) = self.__discipline.io.input_grammar.data_converter.compute_names_to_slices(
            self.differentiated_input_names_substitute,
            input_data,
            self.__input_names_to_sizes,
        )

    def __create_discipline_input_data(
        self,
        x_vect: ndarray,
    ) -> dict[str, ndarray]:
        """Create the discipline input data from the function input vector.

        The variables in the input data are cast according to the types defined in the
        design space.

        Args:
            x_vect: The input vector of the function.

        Returns:
            The input data of the discipline.
        """
        if self.__default_inputs:
            self.__discipline.io.input_grammar.defaults.update(self.__default_inputs)

        if not self.__input_names_to_slices:
            self.__create_input_names_to_slices()

        input_data = (
            self.__discipline.io.input_grammar.data_converter.convert_array_to_data(
                x_vect, self.__input_names_to_slices
            )
        )
        variable_types = x_vect.dtype.metadata
        if variable_types is not None:
            # Restore the proper data types as declared in the design space.
            for name, type_ in variable_types.items():
                input_data[name] = input_data[name].astype(type_, copy=False)

        return input_data
