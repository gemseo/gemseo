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
"""A collection of train-test splits."""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemseo.mlearning.resampling.split import Split


class Splits(Collection):
    """A collection of train-test splits."""

    __splits: tuple[Split]
    """The train-test splits."""

    def __init__(self, *splits: Split) -> None:
        """
        Args:
            *splits: The train-test splits.
        """  # noqa: D205 D212
        self.__splits = splits

    def __contains__(self, item: Split) -> bool:
        return item in self.__splits

    def __iter__(self) -> Iterator[Split]:
        return self.__splits.__iter__()

    def __len__(self) -> int:
        return len(self.__splits)

    def __eq__(self, other: Splits) -> bool:
        return self.__splits == other.__splits
