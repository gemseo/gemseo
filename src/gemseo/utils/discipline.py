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

"""Discipline utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.discipline.discipline import Discipline

if TYPE_CHECKING:
    from collections.abc import Iterable

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


class DummyDiscipline(Discipline):
    """A dummy discipline that does nothing."""

    def __init__(
        self,
        name: str = "",
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            input_names: The names of the input variables, if any.
            output_names: The names of the output variables, if any.
        """  # noqa: D205 D212 D415
        super().__init__(name=name)
        self.input_grammar.update_from_names(input_names)
        self.output_grammar.update_from_names(output_names)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        pass
