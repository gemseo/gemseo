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
"""A staleness guard shared by the collaborators of a design space that derive data."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class StalenessGuard:
    """A staleness guard keyed by an arbitrary comparable version key."""

    rebuild: Callable[[], None]
    """The callback rebuilding the values."""

    __key: object = field(default=None, init=False, repr=False)
    """The version key at the last refresh (`None` until the first refresh)."""

    def refresh(self, key: object) -> None:
        """Rebuild the values if the version key changed since last refresh.

        Args:
            key: The current version key.
        """
        # None is never equal to a real key, so the first refresh always rebuilds.
        if self.__key != key:
            self.rebuild()
            self.__key = key
