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
"""A dictionary-like interface to store the properties of grammar elements."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.grammars._utils import NOT_IN_THE_GRAMMAR_MESSAGE
from gemseo.core.serializable import Serializable
from gemseo.typing import MutableStrKeyMapping
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self

    from gemseo.core.grammars.base_grammar import BaseGrammar
    from gemseo.typing import StrKeyMapping


# This class cannot derive from dict because dict unpickling is specific,
# it calls __setitem__ before the attribute __grammar exists.
class GrammarProperties(
    Serializable,
    MutableStrKeyMapping,
    metaclass=ABCGoogleDocstringInheritanceMeta,
):
    """A dictionary-like interface to store the properties of grammar elements.

    The namespace settings of the grammar are taken into account.
    """

    __data: MutableStrKeyMapping
    """The internal dict-like object."""

    __grammar: BaseGrammar
    """The grammar bound to the properties."""

    def __init__(
        self,
        grammar: BaseGrammar,
        data: StrKeyMapping,
    ) -> None:
        """
        Args:
            grammar: The grammar bound to the properties.
        """  # noqa: D205, D212, D415
        self.__data = {}
        self.__grammar = grammar
        # Explicitly set the items such that they are checked.
        if data:
            self.update(data)

    def __setitem__(self, name: str, value: Any) -> None:
        if name not in self.__grammar:
            msg = NOT_IN_THE_GRAMMAR_MESSAGE.format(name)
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

    def __copy__(self) -> Self:
        # Bypass the checking done in update since it has already been done.
        obj = self.__class__(self.__grammar, {})
        obj.__data = copy(self.__data)
        return obj

    def copy(self) -> Self:  # noqa: D102
        return self.__copy__()
