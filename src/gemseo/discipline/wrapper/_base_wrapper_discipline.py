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
"""Base wrapper discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.core.grammar.base import BaseGrammar


class BaseWrapperDiscipline(Discipline):
    """A base class for a discipline wrapping another discipline."""

    _discipline: Discipline
    """The wrapped discipline."""

    def __init__(
        self,
        wrapped_discipline: Discipline,
        copy_grammars: bool = False,
    ) -> None:
        """
        Args:
            wrapped_discipline: The wrapped discipline.
            copy_grammars: Whether to copy the grammars of the wrapped discipline.
                Otherwise, an update is performed.

        """  # noqa:D205 D212 D415
        super().__init__(name=f"{self.__class__.__name__}({wrapped_discipline})")

        self._discipline = wrapped_discipline

        if copy_grammars:
            self.input_grammar = self._discipline.input_grammar.copy()
            self.output_grammar = self._discipline.output_grammar.copy()
        else:
            self.input_grammar.update(self._discipline.input_grammar)
            self.output_grammar.update(self._discipline.output_grammar)

        self._differentiated_input_names = self._discipline._differentiated_input_names

        self._differentiated_output_names = (
            self._discipline._differentiated_output_names
        )

    @staticmethod
    def _get_wrapped_name(
        name: str, grammar: BaseGrammar, wrapped_grammar: BaseGrammar
    ) -> str:
        """Return the name of a variable of the wrapped discipline.

        A variable has at most one namespace,
        either set on the wrapped discipline or added on the wrapper:
        the name is kept in the first case
        and stripped of its namespace prefix in the second one.

        Args:
            name: The name of the variable of the wrapper.
            grammar: The grammar of the wrapper.
            wrapped_grammar: The grammar of the wrapped discipline.

        Returns:
            The name of the variable of the wrapped discipline.
        """
        if name in wrapped_grammar:
            return name

        return grammar.from_namespaced.get(name, name)

    def _get_wrapped_input_data(self) -> DisciplineData | dict[str, Any]:
        """Return the input data of the wrapped discipline.

        Returns:
            The input data of the wrapper
            keyed by the names of the wrapped discipline.
        """
        io = self.io
        grammar = io.input_grammar
        if not grammar.from_namespaced:
            return io.input_data

        wrapped_grammar = self._discipline.io.input_grammar
        get_wrapped_name = self._get_wrapped_name
        return {
            get_wrapped_name(name, grammar, wrapped_grammar): value
            for name, value in io.input_data.items()
        }
