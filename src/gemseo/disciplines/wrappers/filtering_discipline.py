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

from gemseo.disciplines.wrappers._base_wrapper_discipline import BaseWrapperDiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.core.discipline import Discipline
    from gemseo.typing import StrKeyMapping


class FilteringDiscipline(BaseWrapperDiscipline):
    """A class to wrap another Discipline for a subset of inputs and outputs."""

    def __init__(
        self,
        discipline: Discipline,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        keep_in: bool = True,  # TODO: API: naming and keep/exclude mechanism
        keep_out: bool = True,  # TODO: API: naming and keep/exclude mechanism
    ) -> None:
        """
        Args:
            input_names: The names of the inputs of interest.
                If empty, use all the inputs.
            output_names: The names of the outputs of interest.
                If empty, use all the outputs.
            keep_in: Whether the provided input names must be kept or excluded.
            keep_out: Whether the provided output names must be kept or excluded.
        """  # noqa:D205 D212 D415
        super().__init__(discipline)

        if input_names:
            if keep_in:
                input_names_to_exclude = self._discipline.io.input_grammar.names - set(
                    input_names
                )
            else:
                input_names_to_exclude = set(input_names)
        else:
            input_names_to_exclude = set()

        for name in input_names_to_exclude:
            del self.io.input_grammar[name]

        self._differentiated_input_names = list(
            set(self._differentiated_input_names) - input_names_to_exclude
        )

        if output_names:
            if keep_out:
                output_names_to_exclude = (
                    self._discipline.io.output_grammar.names - set(output_names)
                )
            else:
                output_names_to_exclude = set(output_names)
        else:
            output_names_to_exclude = set()

        for name in output_names_to_exclude:
            del self.io.output_grammar[name]

        self._differentiated_output_names = list(
            set(self._differentiated_output_names) - output_names_to_exclude
        )

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        # TODO: try to use _run instead of execute
        self._discipline.execute(input_data)

        output_grammar = self.io.output_grammar
        return {
            name: value
            for name, value in self._discipline.get_output_data().items()
            if name in output_grammar
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self._discipline._compute_jacobian(input_names, output_names)

        wrapped_jac = self._discipline.jac
        jac = self.jac
        for output_name in output_names:
            jac_output = jac[output_name] = {}
            wrapped_jac_output = wrapped_jac[output_name]
            for input_name in input_names:
                jac_output[input_name] = wrapped_jac_output[input_name]
