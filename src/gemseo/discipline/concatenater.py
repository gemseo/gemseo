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
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The concatenation of several input variables into a single one."""

from __future__ import annotations

from collections import Counter
from itertools import accumulate
from typing import TYPE_CHECKING

from numpy import concatenate
from scipy.sparse import csr_array

from gemseo.core.discipline import Discipline
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.util.typing import StrKeyMapping


class Concatenater(Discipline):
    """Concatenate input variables into a single output variable.

    These input variables can be scaled before concatenation.
    """

    __input_names: tuple[str, ...]
    """The names of the input variables, in the order of the concatenation."""

    def __init__(
        self,
        input_variables: Sequence[str],
        output_variable: str,
        input_coefficients: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            input_variables: The names of the input variables to concatenate.
                The values are concatenated in the order of these names.
            output_variable: The output variable name.
            input_coefficients: The coefficients
                related to the different input variables.

        Raises:
            ValueError: When an input name is repeated.
        """  # noqa: D205 D212 D415
        repeated_names = {
            name for name, count in Counter(input_variables).items() if count > 1
        }
        if repeated_names:
            msg = f"The input names cannot be repeated: {pretty_repr(repeated_names)}."
            raise ValueError(msg)

        super().__init__()
        self.io.input_grammar.update_from_names(input_variables)
        self.io.output_grammar.update_from_names([output_variable])

        self.__coefficients = dict.fromkeys(input_variables, 1.0)
        if input_coefficients:
            self.__coefficients.update(input_coefficients)

        self.__input_names = tuple(input_variables)
        self.__output_name = output_variable

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {
            self.__output_name: concatenate([
                self.__coefficients[input_name] * input_data[input_name]
                for input_name in self.__input_names
            ])
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Compute the Jacobian with respect to all the inputs and for the output.

        Args:
            input_names: These names are ignored.
            output_names: These names are ignored.
        """
        input_data = self.io.get_input_data(with_namespaces=False)
        input_sizes = tuple(input_data[name].size for name in self.__input_names)
        total_size = sum(input_sizes)
        jac = {}
        for name, size, start in zip(
            self.__input_names,
            input_sizes,
            accumulate(input_sizes, initial=0),
            strict=False,
        ):
            jac[name] = csr_array(
                (
                    [self.__coefficients[name]] * size,
                    (range(start, start + size), range(size)),
                ),
                shape=(total_size, size),
            )
        self.jac = {self.__output_name: jac}
