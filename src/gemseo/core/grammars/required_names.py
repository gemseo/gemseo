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
"""Grammar required names."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import MutableSet
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from typing_extensions import Self

    from gemseo.core.grammars.base_grammar import BaseGrammar


class RequiredNames(MutableSet[str]):
    """A set-like class for handling grammar required names.

    The names in an instance of this class shall belong to the bound grammar.
    """

    __grammar: BaseGrammar
    """The grammar bound to the required names."""

    __names: set[str]
    """The required names."""

    def __init__(
        self,
        grammar: BaseGrammar,
        names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            grammar: The grammar bound to the required names.
            names: The required names.
        """  # noqa: D205, D212, D415
        self.__grammar = grammar
        self.__names = set(names)
        # Do not use the names variable such that it works with generators
        # (used by __sub__).
        self.__grammar._check_name(*self.__names)

    def add(self, name: str) -> None:  # noqa: D102
        self.__grammar._check_name(name)
        self.__names.add(name)

    def __contains__(self, name: Any) -> bool:
        return name in self.__names

    def __iter__(self) -> Iterator[str]:
        return iter(self.__names)

    def __len__(self) -> int:
        return len(self.__names)

    def discard(self, name: str) -> None:  # noqa: D102
        self.__names.discard(name)

    def __str__(self) -> str:
        return str(self.__names)

    def _from_iterable(self, names: Iterable[str]) -> Self:
        # This method is required for propagating the bound grammar
        # for operations that imply other sets that are not instances
        # of this class (like - or | or &).
        return self.__class__(self.__grammar, names)
