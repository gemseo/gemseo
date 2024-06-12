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
# Antoine DECHAUME
"""Provide a dict-like class for storing disciplines data."""

from __future__ import annotations

from pathlib import Path
from pathlib import PurePath
from typing import TYPE_CHECKING
from typing import Any

from gemseo.utils.portable_path import to_os_specific

if TYPE_CHECKING:
    from gemseo import StrKeyMapping


class DisciplineData(dict):
    """A dict-like class for handling disciplines data."""

    def __getstate__(self) -> dict[str, Any]:
        state = self.copy()
        for item_name, item_value in self.items():
            if isinstance(item_value, Path):
                # This is needed to handle the case where serialization and
                # deserialization are not made on the same platform.
                state[item_name] = to_os_specific(item_value)
        return state

    def __setstate__(
        self,
        state: StrKeyMapping,
    ) -> None:
        self.update(state)
        for item_name, item_value in state.items():
            if isinstance(item_value, PurePath):
                self[item_name] = Path(item_value)
