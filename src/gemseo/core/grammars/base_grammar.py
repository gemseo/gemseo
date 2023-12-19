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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class for validating data structures."""

from __future__ import annotations

import collections
import logging
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import KeysView
from collections.abc import Mapping
from collections.abc import MutableMapping
from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Optional

from gemseo.core.data_converters.factory import DataConverterFactory
from gemseo.core.grammars.defaults import Defaults
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.namespaces import NamespacesMapping
from gemseo.core.namespaces import namespaces_separator
from gemseo.core.namespaces import update_namespaces
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

NamesToTypes = Mapping[str, Optional[type]]

if TYPE_CHECKING:
    from typing_extensions import Self

    from gemseo.core.data_converters.base import BaseDataConverter
    from gemseo.core.discipline_data import Data
    from gemseo.core.grammars.simple_grammar import SimpleGrammar

LOGGER = logging.getLogger(__name__)


class BaseGrammar(collections.abc.Mapping, metaclass=ABCGoogleDocstringInheritanceMeta):
    """An abstract base class for grammars with a dictionary-like interface.

    A grammar considers a certain type of data defined by mandatory and optional names
    bound to types. A name-type pair is referred to as a grammar *element*. A grammar
    can validate a data from these elements.
    """

    name: str
    """The name of the grammar."""

    to_namespaced: NamespacesMapping
    """The mapping from element names without namespace prefix to element names with
    namespace prefix."""

    from_namespaced: NamespacesMapping
    """The mapping from element names with namespace prefix to element names without
    namespace prefix."""

    _defaults: Defaults
    """The mapping from the names to the default values, if any."""

    _data_converter: BaseDataConverter
    """The converter of data values to NumPy arrays and vice-versa."""

    DATA_CONVERTER_CLASS: ClassVar[str | type[BaseDataConverter]]
    """The class or the class name of the data converter."""

    def __init__(
        self,
        name: str,
    ) -> None:
        """
        Args:
            name: The name of the grammar.

        Raises:
            ValueError: If the name is empty.
        """  # noqa: D205, D212, D415
        if not name:
            raise ValueError("The grammar name cannot be empty.")
        self.name = name
        self.clear()
        self.__create_data_converter(self.DATA_CONVERTER_CLASS)

    def __str__(self) -> str:
        return f"Grammar name: {self.name}"

    @property
    def __string_representation(self) -> MultiLineString:
        """The string representation of the grammar."""
        text = MultiLineString()
        text.add(str(self))
        text.indent()
        text.add("Required elements:")
        text.indent()
        self.__update_grammar_repr(text, True)
        text.dedent()
        text.add("Optional elements:")
        text.indent()
        self.__update_grammar_repr(text, False)
        return text

    def __repr__(self) -> str:
        return str(self.__string_representation)

    def _repr_html_(self) -> str:
        return self.__string_representation._repr_html_()

    def __delitem__(
        self,
        name: str,
    ) -> None:
        self._check_name(name)
        self._defaults.pop(name, None)
        self._delitem(name)

    @abstractmethod
    def _delitem(self, name: str) -> None:
        """Remove an element but the defaults.

        Args:
            name: The name of the element to remove.
        """

    def __copy__(self) -> Self:
        """Create a shallow copy.

        Returns:
            The shallow copy.
        """
        grammar = self.__class__(self.name)
        grammar.to_namespaced = copy(self.to_namespaced)
        grammar.from_namespaced = copy(self.from_namespaced)
        self._copy(grammar)
        grammar._defaults.update(self._defaults)
        return grammar

    copy = __copy__

    @abstractmethod
    def _copy(self, grammar: BaseGrammar) -> None:
        """Copy the specific attribute of a derived class.

        Args:
            grammar: The grammar to be copied into.
        """

    def __update_grammar_repr(self, repr_: MultiLineString, required: bool) -> None:
        """Update the string representation of the grammar with that of its elements.

        Args:
            repr_: The string representation of the grammar.
            required: Whether to show the required elements or the other ones.
        """
        for name, properties in self.items():
            if (name in self.required_names) == required:
                repr_.add(f"{name}:")
                repr_.indent()
                self._update_grammar_repr(repr_, properties)
                if not required:
                    repr_.add(f"Default: {self._defaults.get(name, 'N/A')}")
                repr_.dedent()

    @abstractmethod
    def _update_grammar_repr(self, repr_: MultiLineString, properties: Any) -> None:
        """Update the string representation of the grammar with an element.

        Args:
            repr_: The string representation of the grammar.
            properties: The properties of the element.
        """

    @property
    def names(self) -> KeysView[str]:
        """The names of the elements."""
        return self.keys()

    @property
    def defaults(self) -> Defaults:
        """The mapping from the names to the default values, if any."""
        return self._defaults

    @defaults.setter
    def defaults(self, data: MutableMapping[str, Any]) -> None:
        self._defaults = Defaults(self, data)

    def clear(self) -> None:
        """Empty the grammar."""
        self.to_namespaced = {}
        self.from_namespaced = {}
        self._defaults = Defaults(self, {})
        self._clear()

    @abstractmethod
    def _clear(self) -> None:
        """Empty the grammar but the defaults and namespace mappings."""

    # TODO: API: rename exclude_names (starts with verb like method) to excluded_names.
    @abstractmethod
    def update(
        self,
        grammar: BaseGrammar,
        exclude_names: Iterable[str] = (),
    ) -> None:
        """Update the grammar from another grammar.

        Args:
            grammar: The grammar to update from.
            exclude_names: The names of the elements that shall not be updated.
        """
        self._update_namespaces_from_grammar(grammar)
        self._defaults.update(grammar._defaults, exclude=exclude_names)

    @abstractmethod
    def update_from_types(
        self,
        names_to_types: NamesToTypes,
        merge: bool = False,
    ) -> None:
        """Update the grammar from names bound to types.

        The updated elements are required.

        Args:
            names_to_types: The mapping defining the data names as keys,
                and data types as values.
            merge: Whether to merge or update the grammar.
        """

    def update_from_data(
        self,
        data: Data,
    ) -> None:
        """Update the grammar from name-value pairs.

        The updated elements are required.

        Args:
            data: The data from which to get the names and types,
                typically ``{element_name: element_value}``.
        """
        if not data:
            return
        self.update_from_types({name: type(value) for name, value in data.items()})

    @abstractmethod
    def update_from_names(
        self,
        names: Iterable[str],
    ) -> None:
        """Update the grammar from names.

        The updated elements are required and bind the names to Numpy arrays.

        Args:
            names: The names to update from.
        """

    def validate(
        self,
        data: Data,
        raise_exception: bool = True,
    ) -> None:
        """Validate data against the grammar.

        Args:
            data: The data to be checked,
                with a dictionary-like format: ``{element_name: element_value}``.
            raise_exception: Whether to raise an exception when the validation fails.

        Raises:
            InvalidDataError: If the validation fails and ``raise_exception`` is
                ``True``.
        """
        error_message = MultiLineString()
        error_message.add(f"Grammar {self.name}: validation failed.")

        missing_names = self.required_names.difference(data)
        if missing_names:
            error_message.add(f"Missing required names: {pretty_str(missing_names)}.")
            data_is_valid = False
        else:
            data_is_valid = self._validate(data, error_message)

        if not data_is_valid:
            LOGGER.error(error_message)
            if raise_exception:
                raise InvalidDataError(str(error_message)) from None

    @abstractmethod
    def _validate(
        self,
        data: Data,
        error_message: MultiLineString,
    ) -> bool:
        """Validate data but for the required names.

        Args:
            data: The data to be checked.
            error_message: The error message.

        Returns:
            Whether the validation passed.
        """

    @property
    def data_converter(self) -> BaseDataConverter:
        """The converter of data values to NumPy arrays and vice versa."""
        return self._data_converter

    # TODO: API: remove in favor of is_numeric?
    @abstractmethod
    def is_array(
        self,
        name: str,
        numeric_only: bool = False,
    ) -> bool:
        """Check if an element is an array.

        Args:
            name: The name of the element.
            numeric_only: Whether to check if the array elements are numbers.

        Returns:
            Whether the element is an array.

        Raises:
            KeyError: If the element is not in the grammar.
        """

    @abstractmethod
    def to_simple_grammar(self) -> SimpleGrammar:
        """Convert the grammar to a :class:`.SimpleGrammar`.

        Returns:
            A :class:`.SimpleGrammar` version of the current grammar.
        """

    def restrict_to(
        self,
        names: Iterable[str],
    ) -> None:
        """Restrict the grammar to the given names.

        Args:
            names: The names of the elements to restrict the grammar to.

        Raises:
            KeyError: If a name is not in the grammar.
        """
        self._check_name(*names)
        self._defaults.restrict(*names)
        self._restrict_to(names)

    @abstractmethod
    def _restrict_to(
        self,
        names: Iterable[str],
    ) -> None:
        """Restrict the grammar to the given names but for the defaults.

        Args:
            names: The names of the elements to restrict the grammar to.
        """

    @property
    @abstractmethod
    def required_names(self) -> set[str]:
        """The names of the required elements."""

    def rename_element(self, current_name: str, new_name: str) -> None:
        """Rename an element.

        Args:
            current_name: The current name of the element.
            new_name: The new name of the element.
        """
        self._check_name(current_name)
        self._rename_element(current_name, new_name)
        self._defaults.rename(current_name, new_name)

    @abstractmethod
    def _rename_element(self, current_name: str, new_name: str) -> None:
        """Rename an element without checking its name and ignoring the defaults.

        Args:
            current_name: The current name of the element.
            new_name: The new name of the element.
        """

    @abstractmethod
    def _check_name(self, *names: str) -> None:
        """Check that the names of elements are valid.

        Args:
            *names: The names to be checked.

        Raises:
            KeyError: If a name is not valid.
        """

    def _update_namespaces_from_grammar(self, grammar: BaseGrammar) -> None:
        """Update the namespaces according to another grammar namespaces.

        Args:
            grammar: The grammar to update from.
        """
        if grammar.to_namespaced:
            update_namespaces(self.to_namespaced, grammar.to_namespaced)
        if grammar.from_namespaced:
            update_namespaces(self.from_namespaced, grammar.from_namespaced)

    def add_namespace(self, name: str, namespace: str) -> None:
        """Add a namespace prefix to an existing grammar element.

        The updated element name will be
        ``namespace``+:data:`~gemseo.core.namespaces.namespace_separator`+``name``.

        Args:
            name: The element name to rename.
            namespace: The name of the namespace.
        """
        self._check_name(name)

        if namespaces_separator in name:
            raise ValueError(f"Variable {name} has already a namespace.")

        new_name = namespace + namespaces_separator + name
        self.rename_element(name, new_name)
        self.to_namespaced[name] = new_name
        self.from_namespaced[new_name] = name

    def __create_data_converter(
        self,
        cls: type[BaseDataConverter] | str,
    ) -> None:
        """Create the data converter.

        Args:
            cls: The class or the class name of the data
        """
        if isinstance(cls, str):
            cls = DataConverterFactory().get_class(cls)
        self._data_converter = cls(grammar=self)
