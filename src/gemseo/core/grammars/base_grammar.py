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
from copy import copy
from typing import Any
from typing import Iterable
from typing import KeysView
from typing import Mapping
from typing import MutableMapping
from typing import Optional
from typing import TYPE_CHECKING

from gemseo.core.discipline_data import Data
from gemseo.core.grammars.defaults import Defaults
from gemseo.core.namespaces import namespaces_separator
from gemseo.core.namespaces import NamespacesMapping
from gemseo.core.namespaces import update_namespaces
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.string_tools import MultiLineString

NamesToTypes = Mapping[str, Optional[type]]

if TYPE_CHECKING:
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

    def __str__(self) -> str:
        return f"Grammar name: {self.name}"

    def __repr__(self) -> str:
        text = MultiLineString()
        text.add(str(self))
        text.indent()
        text.add("Required elements:")
        text.indent()
        self._repr_required_elements(text)
        text.dedent()
        text.add("Optional elements:")
        text.indent()
        self._repr_optional_elements(text)
        return str(text)

    @abstractmethod
    def _repr_required_elements(self, text: MultiLineString) -> None:
        """Represent the required elements for `__repr__`.

        Args:
            text: The text to be updated.
        """

    @abstractmethod
    def _repr_optional_elements(self, text: MultiLineString) -> None:
        """Represent the optional elements for `__repr__`.

        Args:
            text: The text to be updated.
        """

    def __delitem__(
        self,
        name: str,
    ) -> None:
        self._defaults.pop(name, None)

    def _copy_base(self) -> BaseGrammar:
        """Create a shallow copy with the attributes of the base class.

        This method is intended to be called from the method that performs a shallow copy
        of a derived class.

        Returns:
            The shallow copy.
        """
        grammar = self.__class__(self.name)
        grammar.to_namespaced = copy(self.to_namespaced)
        grammar.from_namespaced = copy(self.from_namespaced)
        grammar._defaults = self._defaults.copy()
        return grammar

    @abstractmethod
    def __copy__(self) -> BaseGrammar:
        """Create a shallow copy.

        Returns:
            The shallow copy.
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

    @abstractmethod
    def update_from_types(
        self,
        names_to_types: NamesToTypes,
        merge: bool = False,
    ) -> None:
        """Update the grammar from names bound to types.

        Args:
            names_to_types: The mapping defining the data names as keys,
                and data types as values.
            merge: Whether to merge or update the grammar.
        """

    @abstractmethod
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
            Whether the element is an array. If `check_items_number` is set to `True`,
            then return whether the element is an array and its items are numbers.

        Raises:
            KeyError: If the element is not in the grammar.
        """

    @abstractmethod
    def to_simple_grammar(self) -> SimpleGrammar:
        """Convert the grammar to a :class:`.SimpleGrammar`.

        Returns:
            A :class:`.SimpleGrammar` version of the current grammar.
        """

    @abstractmethod
    def update_from_data(
        self,
        data: Data,
    ) -> None:
        """Update the grammar from name-value pairs.

        Args:
            data: The data from which to get the names and types,
                typically ``{element_name: element_value}``.

        Raises:
            TypeError: If a value has a bad type.
        """

    @abstractmethod
    def update_from_names(
        self,
        names: Iterable[str],
    ) -> None:
        """Update the grammar from names.

        The updated elements of the grammar will bind the names to Numpy arrays.

        Args:
            names: The names to update from.
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
        self._defaults.restrict(*names)

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
        self._defaults.rename(current_name, new_name)

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
