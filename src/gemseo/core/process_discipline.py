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
"""Process discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core._execution_status_observer import DisciplinesStatusObserver
from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline.base_discipline import BaseDiscipline


class ProcessDiscipline(Discipline):
    """A discipline that is also a system of disciplines."""

    default_grammar_type = Discipline.GrammarType.SIMPLER

    __disciplines: tuple[BaseDiscipline, ...]
    """The disciplines."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[BaseDiscipline],
        name: str = "",
    ) -> None:
        Discipline.__init__(self, name)
        self.__disciplines = tuple(disciplines)
        if self.cache is not None:
            self.cache._post_set_tolerance = self._post_set_cache_tolerance
        self.execution_status.add_observer(
            DisciplinesStatusObserver(self.__disciplines)
        )

    @property
    def disciplines(self) -> tuple[BaseDiscipline, ...]:
        """The disciplines."""
        return self.__disciplines

    def _post_set_cache_tolerance(self) -> None:
        """Propagate a cache tolerance change to the sub-disciplines."""
        for discipline in self.__disciplines:
            if discipline.cache is not None:
                discipline.cache.tolerance = self.cache.tolerance
