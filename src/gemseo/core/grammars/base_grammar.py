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

import logging
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Optional

from gemseo.core.data_converters.factory import DataConverterFactory
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.grammar_properties import GrammarProperties
from gemseo.core.grammars.required_names import RequiredNames
from gemseo.core.namespaces import MutableNamespacesMapping
from gemseo.core.namespaces import namespaces_separator
from gemseo.core.namespaces import update_namespaces
from gemseo.typing import StrKeyMapping
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import KeysView
    from collections.abc import Mapping

    from typing_extensions import Self

    from gemseo.core.data_converters.base import BaseDataConverter
    from gemseo.core.grammars.simple_grammar import SimpleGrammar

    SimpleGrammarTypes = Mapping[str, Optional[type[Any]]]

LOGGER = logging.getLogger(__name__)


class BaseGrammar(
    StrKeyMapping,
    metaclass=ABCGoogleDocstringInheritanceMeta,
):
    """An abstract base class for grammars with a dictionary-like interface.

    A grammar considers a certain type of data defined by mandatory and optional names
    bound to types. A name-type pair is referred to as a grammar *element*. A grammar
    can validate a data from these elements.

    The names can include any character
    except the special character :attr:`.namespaces_separator` (default: ``":"``)

    Notes:
        Contrary to the standard dictionary, the :meth:``.copy`` method creates
        a deep copy.

    Warnings:
        A name can be prefixed by a namespace.
        Never add a namespace by any other means
        than the method :meth:`.add_namespace`.
    """

    name: str
    """The name of the grammar."""

    to_namespaced: MutableNamespacesMapping
    """The mapping from element names without namespace prefix to element names with
    namespace prefix."""

    from_namespaced: MutableNamespacesMapping
    """The mapping from element names with namespace prefix to element names without
    namespace prefix."""

    _defaults: GrammarProperties
    """The mapping from the names to the default values, if any."""

    _descriptions: GrammarProperties
    """The mapping from the names to the descriptions."""

    _data_converter: BaseDataConverter[BaseGrammar]
    """The converter of data values to NumPy arrays and vice-versa."""

    _required_names: RequiredNames
    """The names of the required elements."""

    DATA_CONVERTER_CLASS: ClassVar[str | type[BaseDataConverter[BaseGrammar]]]
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
            msg = "The grammar name cannot be empty."
            raise ValueError(msg)
        self.name = name
        self.clear()
        self.__create_data_converter(self.DATA_CONVERTER_CLASS)

    def __str__(self) -> str:
        return f"Grammar name: {self.name}"

    def __string_representation(self) -> MultiLineString:
        """Return the string representation of the grammar.

        Returns:
            The string representation of the grammar.
        """
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
        return str(self.__string_representation())

    def _repr_html_(self) -> str:
        return self.__string_representation()._repr_html_()

    def __delitem__(
        self,
        name: str,
    ) -> None:
        self._check_name(name)
        for properties in (self._defaults, self._descriptions):
            properties.pop(name, None)

        self._required_names.discard(name)
        self._delitem(name)

    @abstractmethod
    def _delitem(self, name: str) -> None:
        """Remove an element but the defaults.

        Args:
            name: The name of the element to remove.
        """

    def __copy__(self) -> Self:
        """Create a deep copy.

        Returns:
            The copy.
        """
        grammar = self.__class__(self.name)
        grammar.to_namespaced = deepcopy(self.to_namespaced)
        grammar.from_namespaced = deepcopy(self.from_namespaced)
        self._copy(grammar)
        grammar._required_names = RequiredNames(grammar, names=self._required_names)
        # Cast to dict to avoid the cost of deep-copying the grammar.
        grammar._defaults = GrammarProperties(grammar, deepcopy(dict(self._defaults)))
        grammar._descriptions = GrammarProperties(
            grammar, deepcopy(dict(self._descriptions))
        )
        return grammar

    copy = __copy__

    @abstractmethod
    def _copy(self, grammar: Self) -> None:
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
            if (name in self._required_names) == required:
                repr_.add(f"{name}:")
                repr_.indent()
                if (description := self._descriptions.get(name)) is not None:
                    repr_.add(f"Description: {description}")

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
    def names_without_namespace(self) -> Iterator[str]:
        """The names of the elements without namespace prefixes."""
        from_namespaced_get = self.from_namespaced.get
        return (from_namespaced_get(name, name) for name in self)

    def has_names(self, names: Iterable[str]) -> bool:
        """Return whether names are all element names.

        Args:
            names: The names to check.

        Returns:
            Whether the names are all element names.
        """
        return set(self.keys()).issuperset(names)

    @property
    def defaults(self) -> GrammarProperties:
        """The mapping from the names to the default values, if any."""
        return self._defaults

    @defaults.setter
    def defaults(self, data: StrKeyMapping) -> None:
        self._defaults = GrammarProperties(self, data)

    @property
    def descriptions(self) -> GrammarProperties:
        """The mapping from the names to the descriptions."""
        return self._descriptions

    @descriptions.setter
    def descriptions(self, data: StrKeyMapping) -> None:
        self._descriptions = GrammarProperties(self, data)

    @property
    def required_names(self) -> RequiredNames:
        """The names of the required elements."""
        return self._required_names

    def clear(self) -> None:
        """Empty the grammar."""
        # _clear shall be called first because it creates specific attributes
        # of derived classes that may be used by the next statements.
        self._clear()
        self.to_namespaced = {}
        self.from_namespaced = {}
        self._defaults = GrammarProperties(self, {})
        self._descriptions = GrammarProperties(self, {})
        self._required_names = RequiredNames(self)

    @abstractmethod
    def _clear(self) -> None:
        """Empty specifically the grammar but the common attributes."""

    def update(
        self,
        grammar: Self,
        excluded_names: Iterable[str] = (),
        merge: bool = False,
    ) -> None:
        """Update the grammar from another grammar.

        If ``grammar`` has namespaces, they will be added to the current grammar.

        Args:
            grammar: The grammar to update from.
            excluded_names: The names of the elements that shall not be updated.
            merge: Whether to merge or update the grammar.
        """
        if not grammar:
            return
        self._update(grammar, excluded_names, merge)
        self.__update_namespaces_from_grammar(grammar)
        for properties, other_properties in (
            (self._defaults, grammar._defaults),
            (self._descriptions, grammar._descriptions),
        ):
            properties.update({
                k: v for k, v in other_properties.items() if k not in excluded_names
            })

        self._required_names |= (grammar.keys() - excluded_names).intersection(
            grammar._required_names.get_names_difference(excluded_names)
        )

    @abstractmethod
    def _update(
        self,
        grammar: Self,
        excluded_names: Iterable[str],
        merge: bool,
    ) -> None:
        """Update specifically the grammar from another grammar.

        Args:
            grammar: The grammar to update from.
            excluded_names: The names of the elements that shall not be updated.
            merge: Whether to merge or update the grammar.
        """

    def update_from_types(
        self,
        names_to_types: SimpleGrammarTypes,
        merge: bool = False,
    ) -> None:
        """Update the grammar from names bound to types.

        The updated elements are required.

        Args:
            names_to_types: The mapping defining the data names as keys,
                and data types as values.
            merge: Whether to merge or update the grammar.
        """
        if not names_to_types:
            return
        self._update_from_types(names_to_types, merge)
        self._required_names |= names_to_types.keys()

    @abstractmethod
    def _update_from_types(
        self,
        names_to_types: SimpleGrammarTypes,
        merge: bool,
    ) -> None:
        """Update specifically the grammar from names bound to types.

        Args:
            names_to_types: The mapping defining the data names as keys,
                and data types as values.
            merge: Whether to merge or update the grammar.
        """

    def update_from_data(
        self,
        data: StrKeyMapping,
        merge: bool = False,
    ) -> None:
        """Update the grammar from name-value pairs.

        The updated elements are required.

        Args:
            data: The data from which to get the names and types,
                typically ``{element_name: element_value}``.
            merge: Whether to merge or update the grammar.
        """
        if not data:
            return
        self._update_from_data(data, merge)
        self._required_names |= data.keys()

    def _update_from_data(
        self,
        data: StrKeyMapping,
        merge: bool,
    ) -> None:
        """Update specifically the grammar from name-value pairs.

        The updated elements are required.

        Args:
            data: The data from which to get the names and types,
                typically ``{element_name: element_value}``.
            merge: Whether to merge or update the grammar.
        """
        self._update_from_types(
            {name: type(value) for name, value in data.items()}, merge=merge
        )

    def update_from_names(
        self,
        names: Iterable[str],
        merge: bool = False,
    ) -> None:
        """Update the grammar from names.

        The updated elements are required and bind the names to NumPy arrays.

        Args:
            names: The names to update from.
            merge: Whether to merge or update the grammar.
        """
        if not names:
            return
        self._update_from_names(names, merge)
        self._required_names |= set(names)

    @abstractmethod
    def _update_from_names(
        self,
        names: Iterable[str],
        merge: bool,
    ) -> None:
        """Update specifically the grammar from names.

        Args:
            names: The names to update from.
            merge: Whether to merge or update the grammar.
        """

    def validate(
        self,
        data: StrKeyMapping,
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

        missing_names = self._required_names.get_names_difference(data)
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
        data: StrKeyMapping,
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
    def data_converter(self) -> BaseDataConverter[BaseGrammar]:
        """The converter of data values to NumPy arrays and vice versa."""
        return self._data_converter

    def to_simple_grammar(self) -> SimpleGrammar:
        """Convert the grammar to a :class:`.SimpleGrammar`.

        .. warning::
            Types in simple grammars are not as precise as for other grammars.
            Thus data that are invalid with a given grammar may be valid
            with a simple grammar obtained from the current method.

        Returns:
            A :class:`.SimpleGrammar` version of the current grammar.
        """
        from gemseo.core.grammars.simple_grammar import SimpleGrammar

        grammar = SimpleGrammar(
            self.name,
            names_to_types=self._get_names_to_types(),
            required_names=self._required_names,
        )
        grammar.defaults = self._defaults
        grammar.descriptions = self._descriptions
        return grammar

    @abstractmethod
    def _get_names_to_types(self) -> SimpleGrammarTypes:
        """Create the mapping from element names to elements types.

        The elements for which types definitions cannot be expressed as a unique Python
        type, the type is set to ``None``.

        Returns:
            The mapping from element names to elements types.
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
        for properties in (self._defaults, self._descriptions):
            for name in properties.keys() - names:
                del properties[name]

        self._required_names &= set(names)
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

    def rename_element(self, current_name: str, new_name: str) -> None:
        """Rename an element.

        Args:
            current_name: The current name of the element.
            new_name: The new name of the element.
        """
        self._check_name(current_name)
        self._rename_element(current_name, new_name)
        if current_name in self._required_names:
            self._required_names.remove(current_name)
            self._required_names.add(new_name)

        for properties in (self._defaults, self._descriptions):
            default_value = properties.pop(current_name, None)
            if default_value is not None:
                properties[new_name] = default_value

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

    def __update_namespaces_from_grammar(self, grammar: Self) -> None:
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

        For example,
        the grammar element named ``"foo"`` will be renamed to ``"ns:foo"
        when using the namespace named ``"ns"``.
        ``":"`` is the default value
        of the special character :attr:`.namespaces_separator`.

        Args:
            name: The element name to rename.
            namespace: The name of the namespace.

        Raises:
            ValueError: If the element already has a namespace prefix.

        Warnings:
           Never add a namespace by any other means
           than the method :meth:`.add_namespace`,
           e.g. do not write
           ``my_discipline.io.input_grammar.update_from_names(["ns:foo"])``,
           but write instead:

            .. code-block:: python

               my_discipline.io.input_grammar.update_from_names(["foo"])
               my_discipline.io.input_grammar.add_namespace("foo", "ns")``.
        """
        self._check_name(name)

        if namespaces_separator in name:
            original_name = self.from_namespaced[name]
            ns = name.strip(original_name).strip(namespaces_separator)
            msg = f"The variable {original_name!r} already has a namespace ({ns!r})."
            raise ValueError(msg)

        new_name = namespace + namespaces_separator + name
        self.rename_element(name, new_name)
        self.to_namespaced[name] = new_name
        self.from_namespaced[new_name] = name

    def __create_data_converter(
        self,
        cls: type[BaseDataConverter[BaseGrammar]] | str,
    ) -> None:
        """Create the data converter.

        Args:
            cls: The class or the class name of the data
        """
        if isinstance(cls, str):
            cls = DataConverterFactory().get_class(cls)
        self._data_converter = cls(grammar=self)
