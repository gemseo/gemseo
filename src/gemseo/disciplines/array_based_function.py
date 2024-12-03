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
"""A discipline wrapping an array-based Python function."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from numpy import zeros

from gemseo.core.discipline.discipline import Discipline
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


class ArrayBasedFunctionDiscipline(Discipline):
    """A discipline wrapping an array-based function.

    Both the unique argument of this Python function and its return value are NumPy
    arrays.
    """

    __function: Callable[[RealArray], RealArray]
    """The function of interest."""

    __jac_function: Callable[[RealArray], RealArray] | None
    """The function computing the derivatives of the function of interest, if any."""

    __variable_names_to_sizes: dict[str, int]
    """The mapping from the names to the sizes for the variables."""

    __output_names_to_sizes: dict[str, int]
    """The mapping from the names to the sizes for the output."""

    def __init__(
        self,
        function: Callable[[RealArray], RealArray],
        input_names_to_sizes: dict[str, int],
        output_names_to_sizes: dict[str, int],
        jac_function: Callable[[RealArray], RealArray] | None = None,
    ):
        """
        Args:
            function: The function of interest
                whose both the unique argument and the output are NumPy arrays.
            input_names_to_sizes: The mapping from the names to the sizes for the input.
            output_names_to_sizes: The mapping
                from the names to the sizes for the output.
            jac_function: The function
                computing the derivatives of the function of interest;
                both its unique argument and its output (i.e. a Jacobian matrix)
                are NumPy arrays.
                If ``None``, the derivatives will have to be approximated if required.
        """  # noqa: D205, D212
        super().__init__()
        self.__function = function
        self.io.input_grammar.update_from_names(input_names_to_sizes)
        self.io.output_grammar.update_from_names(output_names_to_sizes)
        self.io.input_grammar.defaults = {
            name: zeros(size) for name, size in input_names_to_sizes.items()
        }
        self.__output_names_to_sizes = output_names_to_sizes.copy()
        self.__variable_names_to_sizes = output_names_to_sizes.copy()
        self.__variable_names_to_sizes.update(input_names_to_sizes)
        self.__jac_function = jac_function

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        input_vector = concatenate_dict_of_arrays_to_array(input_data, input_data)
        output_vector = self.__function(input_vector)
        return split_array_to_dict_of_arrays(
            output_vector, self.__output_names_to_sizes, self.io.output_grammar
        )

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Raises:
            RuntimeError: When the discipline cannot compute the analytic derivatives.
        """  # noqa: D205 D212 D415
        if self.__jac_function is None:
            msg = f"The discipline {self.name} cannot compute the analytic derivatives."
            raise RuntimeError(msg)

        input_data = self.io.get_input_data()
        input_vector = concatenate_dict_of_arrays_to_array(
            input_data, self.io.input_grammar
        )
        self.jac = split_array_to_dict_of_arrays(
            self.__jac_function(input_vector),
            self.__variable_names_to_sizes,
            self.io.output_grammar,
            self.io.input_grammar,
        )
