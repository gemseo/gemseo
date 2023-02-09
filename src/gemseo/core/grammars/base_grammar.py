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

import abc
import collections
import logging
from typing import Container
from typing import Iterable
from typing import KeysView
from typing import Sequence
from typing import TYPE_CHECKING

from gemseo.core.discipline_data import Data
from gemseo.core.namespaces import namespaces_separator
from gemseo.core.namespaces import NamespacesMapping
from gemseo.core.namespaces import update_namespaces
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.core.grammars.simple_grammar import SimpleGrammar

LOGGER = logging.getLogger(__name__)


class BaseGrammar(collections.abc.Mapping, metaclass=ABCGoogleDocstringInheritanceMeta):
    """An abstract base class for grammars with a dictionary-like interface.

    A grammar considers a certain type of data defined by mandatory and optional names
    bound to types. A name-type pair is referred to as a grammar *element*. A grammar can
    validate a data from these elements.
    """

    name: str
    """The name of the grammar."""

    to_namespaced: NamespacesMapping
    """The mapping from element names without namespace prefix to element names with
    namespace prefix."""

    from_namespaced: NamespacesMapping
    """The mapping from element names with namespace prefix to element names without
    namespace prefix."""

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
        self.to_namespaced = {}
        self.from_namespaced = {}
        self.clear()

    def __str__(self) -> str:
        return f"Grammar name: {self.name}"

    @property
    def names(self) -> KeysView[str]:
        """The names of the elements."""
        return self.keys()

    @abc.abstractmethod
    def clear(self) -> None:
        """Empty the grammar."""

    @abc.abstractmethod
    def update(
        self,
        grammar: BaseGrammar | Iterable[str],
        exclude_names: Container[str] | None = None,
    ) -> None:
        """Update the grammar.

        Args:
            grammar: The grammar or names to update from.
            exclude_names: The names of the elements that shall not be updated.
        """

    @abc.abstractmethod
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
            InvalidDataException: If the validation fails and ``raise_exception`` is
                ``True``.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def convert_to_simple_grammar(self) -> SimpleGrammar:
        """Convert the grammar to a :class:`.SimpleGrammar`.

        Returns:
            A :class:`.SimpleGrammar` version of the current grammar.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def restrict_to(
        self,
        names: Sequence[str],
    ) -> None:
        """Restrict the grammar to the given names.

        Args:
            names: The names of the elements to restrict the grammar to.

        Raises:
            KeyError: If a name is not in the grammar.
        """

    @property
    @abc.abstractmethod
    def required_names(self) -> set[str]:
        """The names of the required elements."""

    @abc.abstractmethod
    def rename_element(self, current_name: str, new_name: str) -> None:
        """Rename an element.

        Args:
            current_name: The current name of the element.
            new_name: The new name of the element.
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

    @abc.abstractmethod
    def _check_name(self, *names: str) -> None:
        """Check that the names of elements are valid.

        Args:
            *names: The names to be checked.

        Raises:
            KeyError: If a name is not valid.
        """
