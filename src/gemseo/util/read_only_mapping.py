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
"""A picklable read-only view over a mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")


class ReadOnlyMapping(Mapping[_KT, _VT]):
    """A picklable, read-only live view over a mapping.

    Unlike [MappingProxyType][types.MappingProxyType],
    this wrapper is picklable,
    so objects exposing it need no bespoke `__getstate__`/`__setstate__`.
    The backing mapping is held by reference (not copied),
    so mutations of the backing mapping are reflected through the view;
    the values of the backing mapping are not copied and remain mutable;
    the view itself forbids insertion, deletion and update.
    """

    __slots__ = ("_mapping",)

    _mapping: Mapping[_KT, _VT]
    """The backing mapping."""

    def __init__(self, mapping: Mapping[_KT, _VT]) -> None:
        """
        Args:
            mapping: The mapping to expose as read-only.
        """  # noqa: D205, D212
        self._mapping = mapping

    def __getitem__(self, key: _KT) -> _VT:
        return self._mapping[key]

    def __iter__(self) -> Iterator[_KT]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self._mapping)!r})"
