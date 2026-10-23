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

from gemseo.core.discipline import Discipline


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
