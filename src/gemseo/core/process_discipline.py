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

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import NoReturn
from typing import final

from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline.base_discipline import BaseDiscipline
    from gemseo.typing import StrKeyMapping


class ProcessDiscipline(Discipline):
    """A discipline that is also a system of disciplines."""

    default_grammar_type = Discipline.GrammarType.SIMPLER

    _disciplines: tuple[BaseDiscipline, ...]
    """The disciplines."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[BaseDiscipline],
        name: str = "",
    ) -> None:
        super().__init__(name)
        self._disciplines = tuple(disciplines)
        if self.cache is not None:
            self.cache._post_set_tolerance = self._post_set_cache_tolerance

    @property
    def disciplines(self) -> tuple[BaseDiscipline, ...]:
        """The disciplines."""
        return self._disciplines

    def _post_set_cache_tolerance(self) -> None:
        """Propagate a cache tolerance change to the sub-disciplines."""
        for discipline in self._disciplines:
            if discipline.cache is not None:
                discipline.cache.tolerance = self.cache.tolerance

    @final
    def _run(self, input_data: StrKeyMapping) -> NoReturn:
        """This method shall be implemented, implement _execute instead."""
        msg = "Do not use _run for process disciplines, use _execute."
        raise NotImplementedError(msg)

    @abstractmethod
    def _execute(self) -> None:
        pass
