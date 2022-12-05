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

from typing import Iterable

from numpy import eye

from gemseo.core.discipline import MDODiscipline


class Splitter(MDODiscipline):
    """A discipline splitting an input variable.

    Several output variables containing slice of the input variable are extracted.

    Example:
        >>> discipline = Splitter("alpha", {"beta": [0, 1], "delta": [2, 3],
        "gamma": 4})
        >>> discipline.execute({"alpha": array([1.0, 2.0, 3.0, 4.0, 5.0])})
        >>> delta = discipline.local_data["delta"]  # delta = array([3.0, 4.0])
    """

    def __init__(
        self,
        input_name: str,
        output_names_to_input_indices: dict[str, Iterable[int] | int],
    ):
        """
        Args:
            input_name: The name of the input to split.
            output_names_to_input_indices: The input indices associated with the
                output names.
        """  # noqa: D205, D212, D415
        self.__input_name = input_name
        for output_name, input_indices in output_names_to_input_indices.items():
            if isinstance(input_indices, int):
                output_names_to_input_indices[output_name] = [input_indices]
        self.__slicing_structure = output_names_to_input_indices

        super().__init__()
        self.input_grammar.update([input_name])
        self.output_grammar.update(output_names_to_input_indices.keys())

    def _run(self) -> None:
        input_data = self.local_data[self.__input_name]
        for output_name, input_indices in self.__slicing_structure.items():
            self.local_data[output_name] = input_data[input_indices]

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._init_jacobian(with_zeros=True)
        self.jac = {}
        identity = eye(self.local_data[self.__input_name].size)
        for output_name, input_indices in self.__slicing_structure.items():
            self.jac[output_name] = {}
            self.jac[output_name][self.__input_name] = identity[input_indices, :]
