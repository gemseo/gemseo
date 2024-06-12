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
"""The grammar default values."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.serializable import Serializable
from gemseo.typing import MutableStrKeyMapping
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemseo.core.grammars.base_grammar import BaseGrammar
    from gemseo.typing import StrKeyMapping


# This class cannot derive from dict because dict unpickling is specific,
# it calls __setitem__ before the attribute __grammar exists.
class Defaults(
    Serializable,
    MutableStrKeyMapping,
    metaclass=ABCGoogleDocstringInheritanceMeta,
):
    """A class for handling grammar default values.

    A dictionary-like interface to bind grammar names to default values. The namespace
    settings of the grammar are taken into account.
    """

    __data: MutableStrKeyMapping
    """The internal dict-like object."""

    __grammar: BaseGrammar
    """The grammar bound to the defaults."""

    def __init__(
        self,
        grammar: BaseGrammar,
        data: StrKeyMapping,
    ) -> None:
        """
        Args:
            grammar: The grammar bound to the defaults.
        """  # noqa: D205, D212, D415
        self.__data = {}
        self.__grammar = grammar
        # Explicitly set the items such that they are checked.
        if data:
            self.update(data)

    def __setitem__(self, name: str, value: Any) -> None:
        if name not in self.__grammar:
            msg = f"The name {name} is not in the grammar."
            raise KeyError(msg)
        self.__data[name] = value

    def __getitem__(self, key: str) -> Any:
        return self.__data[key]

    def __delitem__(self, key: str) -> None:
        del self.__data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return repr(self.__data)
