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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Filtering wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping


class FilteringDiscipline(MDODiscipline):
    """A class to wrap another MDODiscipline for a subset of inputs and outputs."""

    def __init__(
        self,
        discipline: MDODiscipline,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        keep_in: bool = True,
        keep_out: bool = True,
    ) -> None:
        """
        Args:
            discipline: The original discipline.
            input_names: The names of the inputs of interest.
                If ``None``, use all the inputs.
            output_names: The names of the outputs of interest.
                If ``None``, use all the outputs.
            keep_in: Whether to the inputs of interest.
                Otherwise, remove them.
            keep_out: Whether to the outputs of interest.
                Otherwise, remove them.
        """  # noqa:D205 D212 D415
        self.discipline = discipline
        super().__init__(name=discipline.name)
        original_input_names = discipline.get_input_data_names()
        original_output_names = discipline.get_output_data_names()
        if not input_names:
            input_names = original_input_names
        elif not keep_in:
            input_names = list(set(original_input_names) - set(input_names))

        if not output_names:
            output_names = original_output_names
        elif not keep_out:
            output_names = list(set(original_output_names) - set(output_names))

        self.input_grammar.update_from_names(input_names)
        self.output_grammar.update_from_names(output_names)
        self.default_inputs = self.__filter_inputs(self.discipline.default_inputs)
        removed_inputs = set(original_input_names) - set(input_names)
        diff_inputs = set(self.discipline._differentiated_inputs) - removed_inputs
        self.add_differentiated_inputs(list(diff_inputs))
        removed_outputs = set(original_output_names) - set(output_names)
        diff_outputs = set(self.discipline._differentiated_outputs) - removed_outputs
        self.add_differentiated_outputs(list(diff_outputs))

    def _run(self) -> None:
        self.discipline.execute(self.get_input_data())
        self.store_local_data(**self.__filter_inputs(self.discipline.local_data))
        self.store_local_data(**self.__filter_outputs(self.discipline.local_data))

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self.discipline._compute_jacobian(inputs, outputs)
        self._init_jacobian(inputs, outputs)
        jac = self.discipline.jac
        for output_name in self.get_output_data_names():
            for input_name in self.get_input_data_names():
                self.jac[output_name][input_name] = jac[output_name][input_name]

    def __filter_inputs(self, data: Mapping[str, Any]):
        """Filter a mapping by input names.

        Args:
            data: The original mapping.

        Returns:
            The mapping filtered by input names.
        """
        return {name: data[name] for name in self.get_input_data_names()}

    def __filter_outputs(self, data):
        """Filter a mapping by output names.

        Args:
            data: The original mapping.

        Returns:
            The mapping filtered by output names.
        """
        return {name: data[name] for name in self.get_output_data_names()}
