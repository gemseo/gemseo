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

from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


class FilteringDiscipline(Discipline):
    """A class to wrap another Discipline for a subset of inputs and outputs."""

    def __init__(
        self,
        discipline: Discipline,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        keep_in: bool = True,
        keep_out: bool = True,
    ) -> None:
        """
        Args:
            discipline: The original discipline.
            input_names: The names of the inputs of interest.
                If empty, use all the inputs.
            output_names: The names of the outputs of interest.
                If empty, use all the outputs.
            keep_in: Whether to keep the inputs of interest.
                Otherwise, remove them.
            keep_out: Whether to keep the outputs of interest.
                Otherwise, remove them.
        """  # noqa:D205 D212 D415
        self.discipline = discipline
        super().__init__(name=discipline.name)
        original_input_names = discipline.io.input_grammar
        original_output_names = discipline.io.output_grammar
        if not input_names:
            input_names = original_input_names
        elif not keep_in:
            input_names = list(set(original_input_names) - set(input_names))

        if not output_names:
            output_names = original_output_names
        elif not keep_out:
            output_names = list(set(original_output_names) - set(output_names))

        self.io.input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names(output_names)
        self.io.input_grammar.defaults = self.__filter_inputs(
            self.discipline.io.input_grammar.defaults
        )
        removed_inputs = set(original_input_names) - set(input_names)
        diff_inputs = set(self.discipline._differentiated_input_names) - removed_inputs
        self.add_differentiated_inputs(list(diff_inputs))
        removed_outputs = set(original_output_names) - set(output_names)
        diff_outputs = (
            set(self.discipline._differentiated_output_names) - removed_outputs
        )
        self.add_differentiated_outputs(list(diff_outputs))

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return self.__filter_outputs(self.discipline.execute(self.io.get_input_data()))

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self.discipline._compute_jacobian(input_names, output_names)
        self._init_jacobian(input_names, output_names)
        jac = self.discipline.jac
        for output_name in self.io.output_grammar:
            for input_name in self.io.input_grammar:
                self.jac[output_name][input_name] = jac[output_name][input_name]

    def __filter_inputs(self, data: StrKeyMapping) -> dict[str, Any]:
        """Filter a mapping by input names.

        Args:
            data: The original mapping.

        Returns:
            The mapping filtered by input names.
        """
        return {name: data[name] for name in self.io.input_grammar}

    def __filter_outputs(self, data) -> dict[str, Any]:
        """Filter a mapping by output names.

        Args:
            data: The original mapping.

        Returns:
            The mapping filtered by output names.
        """
        return {name: data[name] for name in self.io.output_grammar}
