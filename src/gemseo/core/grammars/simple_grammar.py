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
"""A basic grammar based on names and types."""

from __future__ import annotations

import collections
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import ndarray

from gemseo.core.grammars._utils import NOT_IN_THE_GRAMMAR_MESSAGE
from gemseo.core.grammars.base_grammar import BaseGrammar

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

    from typing_extensions import Self

    from gemseo.core.grammars.base_grammar import SimpleGrammarTypes
    from gemseo.typing import StrKeyMapping
    from gemseo.utils.string_tools import MultiLineString


class SimpleGrammar(BaseGrammar):
    """A grammar based on names and types with a dictionary-like interface.

    The types are pure Python types, type annotations are not supported.

    The grammar could be empty, in that case the data validation always pass. If the
    type bound to a name is ``None`` then the type of the corresponding data name is
    always valid.

    .. warning:: This grammar cannot merge elements. Merging will raise an error.
    """

    DATA_CONVERTER_CLASS: ClassVar[str] = "SimpleGrammarDataConverter"

    __names_to_types: dict[str, type | None]
    """The mapping from element names to element types."""

    def __init__(
        self,
        name: str,
        names_to_types: SimpleGrammarTypes | None = None,
        required_names: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            names_to_types: The mapping defining the data names as keys,
                and data types as values.
                If ``None``, the grammar is empty.
            required_names: The names of the required elements.
                If ``None``, all the elements are required.
            **kwargs: These arguments are not used.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        if names_to_types:
            self.update_from_types(names_to_types)
        if required_names is not None:
            self._required_names.clear()
            self._required_names |= set(required_names)

    def __getitem__(self, name: str) -> type | None:
        return self.__names_to_types[name]

    def __len__(self) -> int:
        return len(self.__names_to_types)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__names_to_types)

    def _delitem(self, name: str) -> None:  # noqa:D102
        del self.__names_to_types[name]

    def _copy(self, grammar: Self) -> None:  # noqa:D102
        grammar.__names_to_types = deepcopy(self.__names_to_types)

    def _rename_element(self, current_name: str, new_name: str) -> None:  # noqa: D102
        self.__names_to_types[new_name] = self.__names_to_types.pop(current_name)

    def _update(  # noqa: D102
        self,
        grammar: Self,
        excluded_names: Iterable[str],
        merge: bool,
    ) -> None:
        """
        Raises:
            ValueError: When merge is ``True``,
                since it is not supported for :class:`.SimpleGrammar`.
        """  # noqa: D205, D212, D415
        self.__check_merge(merge)
        self.__update(grammar.to_simple_grammar(), excluded_names)

    def _update_from_names(  # noqa: D102
        self,
        names: Iterable[str],
        merge: bool,
    ) -> None:
        """
        Raises:
            ValueError: When merge is ``True``,
                since it is not supported for :class:`.SimpleGrammar`.
        """  # noqa: D205, D212, D415
        self.__check_merge(merge)
        self.__update(dict.fromkeys(names, ndarray))

    def _update_from_types(
        self,
        names_to_types: SimpleGrammarTypes,
        merge: bool,
    ) -> None:
        """
        Raises:
            ValueError: When merge is ``True``,
                since it is not supported for :class:`.SimpleGrammar`.
        """  # noqa: D205, D212, D415
        self.__check_merge(merge)
        self.__update(names_to_types)

    @classmethod
    def __check_merge(cls, merge: bool) -> None:
        """Check that merge is not ``True`` since it is not supported.

        Args:
            merge: Whether to merge or update the grammar.

        Raises:
            ValueError: When merge is ``True``,
                since it is not supported for :class:`.SimpleGrammar`.
        """
        if merge:
            msg = f"Merge is not supported for {cls.__name__}."
            raise ValueError(msg)

    def __update(
        self,
        grammar: Self | StrKeyMapping,
        excluded_names: Iterable[str] = (),
    ) -> None:
        """Update the elements from another grammar or elements.

        When elements are provided names and types instead of a
        :class:`.BaseGrammar`,
        for consistency with :class:`.__init__` behavior
        it is assumed that all of them are required.

        Args:
            grammar: The grammar to take the elements from.
            excluded_names: The names of the elements that shall not be updated.

        Raises:
            ValueError: If the types are bad.
        """
        for element_name, element_type in grammar.items():
            if element_name in excluded_names:
                continue

            self.__check_type(element_name, element_type)

            # Generalize dict to support DisciplineData objects.
            if element_type is dict:
                self.__names_to_types[element_name] = collections.abc.Mapping
            else:
                self.__names_to_types[element_name] = element_type

    def _clear(self) -> None:  # noqa: D102
        self.__names_to_types = {}

    def _update_grammar_repr(self, repr_: MultiLineString, properties: Any) -> None:
        repr_.add(f"Type: {properties}")

    def _validate(  # noqa: D102
        self,
        data: StrKeyMapping,
        error_message: MultiLineString,
    ) -> bool:
        data_is_valid = True
        for element_name, element_type in self.__names_to_types.items():
            if (
                element_name in data
                and element_type is not None
                and not isinstance(data[element_name], element_type)
            ):
                error_message.add(
                    f"Bad type for {element_name}: "
                    f"{type(data[element_name])} instead of {element_type}."
                )
                data_is_valid = False
        return data_is_valid

    def _restrict_to(  # noqa: D102
        self,
        names: Iterable[str],
    ) -> None:
        for element_name in self.__names_to_types.keys() - names:
            del self.__names_to_types[element_name]

    def to_simple_grammar(self) -> Self:  # noqa: D102
        return self

    def _get_names_to_types(self) -> SimpleGrammarTypes:  # pragma: no cover
        # This method is never called but is abstract.
        return self

    @staticmethod
    def __check_type(name: str, obj: Any) -> None:
        """Check that the type of object is a valid element type.

        Raises:
            TypeError: If the object is neither a type nor ``None``.
        """
        if obj is not None and not isinstance(obj, type):
            msg = f"The element {name} must be a type or None: it is {obj}."
            raise TypeError(msg)

    def _check_name(self, *names: str) -> None:
        for name in names:
            if name not in self.__names_to_types:
                msg = NOT_IN_THE_GRAMMAR_MESSAGE.format(name)
                raise KeyError(msg)
