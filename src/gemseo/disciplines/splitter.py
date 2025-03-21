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
"""A discipline splitting an input variable."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import ndarray
from scipy.sparse import eye

from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


class Splitter(Discipline):
    """A discipline splitting an input variable.

    Several output variables containing slice of the input variable are extracted.

    Examples:
        >>> discipline = Splitter("alpha", {"beta": [0, 1], "delta": [2, 3],
        "gamma": 4})
        >>> discipline.execute({"alpha": array([1.0, 2.0, 3.0, 4.0, 5.0])})
        >>> delta = discipline.io.data["delta"]  # delta = array([3.0, 4.0])
    """

    def __init__(
        self,
        input_name: str,
        output_names_to_input_indices: dict[str, Iterable[int] | int],
    ) -> None:
        """
        Args:
            input_name: The name of the input to split.
            output_names_to_input_indices: The input indices associated with the
                output names.
        """  # noqa: D205, D212, D415
        self.__input_name = input_name
        for output_name, input_indices in output_names_to_input_indices.items():
            if not isinstance(input_indices, ndarray) and not isinstance(
                input_indices, list
            ):
                output_names_to_input_indices[output_name] = [input_indices]
        self.__slicing_structure = output_names_to_input_indices

        super().__init__()
        self.io.input_grammar.update_from_names([input_name])
        self.io.output_grammar.update_from_names(output_names_to_input_indices.keys())

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        input_data = input_data[self.__input_name]
        output_data = {}
        for output_name, input_indices in self.__slicing_structure.items():
            output_data[output_name] = input_data[input_indices]
        return output_data

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self._init_jacobian(init_type=self.InitJacobianType.SPARSE)
        identity = eye(self.io.data[self.__input_name].size, format="csr")
        for output_name, input_indices in self.__slicing_structure.items():
            self.jac[output_name][self.__input_name] = identity[input_indices, :]
