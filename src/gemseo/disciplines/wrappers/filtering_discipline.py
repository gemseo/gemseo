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
    from gemseo.core.grammars.base import BaseGrammar
    from gemseo.typing import StrKeyMapping


class FilteringDiscipline(BaseWrapperDiscipline):
    """A class to wrap another Discipline for a subset of inputs and outputs."""

    def __init__(
        self,
        discipline: Discipline,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        use_input_names: bool = True,
        use_output_names: bool = True,
    ) -> None:
        """
        Args:
            input_names: The names of the inputs of interest.
                If empty, use all the inputs.
            output_names: The names of the outputs of interest.
                If empty, use all the outputs.
            use_input_names: Whether to define the input grammar from `input_names`.
                Otherwise, define it from the other inputs of the discipline.
            use_output_names: Whether to define the output grammar from `output_names`.
                Otherwise, define it from the other outputs of the discipline.
        """  # noqa:D205 D212 D415
        super().__init__(discipline)
        self.__set_grammar(
            self.io.input_grammar,
            input_names,
            use_input_names,
            self._differentiated_input_names,
        )
        self.__set_grammar(
            self.io.output_grammar,
            output_names,
            use_output_names,
            self._differentiated_output_names,
        )

    @staticmethod
    def __set_grammar(
        grammar: BaseGrammar,
        variable_names: Iterable[str],
        keep_variable_names: bool,
        differentiated_variable_names: list[str],
    ):
        """Set a grammar.

        Args:
            grammar: The grammar to set.
            variable_names: The names of the variables of interest.
                If empty, use all the variables.
            keep_variable_names: Whether to define the grammar from `variable_names`.
                Otherwise, define it from the other variables of the grammar.
            differentiated_variable_names: The names of the differentiated variables.
        """
        variable_names = set(variable_names)
        if variable_names and keep_variable_names:
            variable_names_to_exclude = grammar.names - variable_names
        else:
            variable_names_to_exclude = variable_names

        for name in variable_names_to_exclude:
            del grammar[name]

        differentiated_variable_names[:] = list(
            set(differentiated_variable_names) - variable_names_to_exclude
        )

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        data = self._discipline.execute(input_data)
        return {name: data[name] for name in self.io.output_grammar}

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
