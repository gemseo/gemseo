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
"""JSON schema handler."""
from __future__ import annotations

from collections import abc
from typing import Any
from typing import Iterator

from genson import SchemaBuilder
from genson.schema.strategies import Object


class _MergeRequiredStrategy(Object):
    """A genson Object with a modified required attribute handling.

    Genson Object does not merge the required attribute on purpose.
    See :ref:`https://github.com/wolverdude/GenSON#genson`.
    This class will merge the required attributes.
    """

    # Do not merge the name and id properties.
    KEYWORDS = Object.KEYWORDS + ("name", "id")

    def add_schema(self, schema) -> None:
        """Add a schema and merge the required attribute.

        Args:
            schema: A schema to be added.
        """
        if "required" not in schema or self._required is None:
            super().add_schema(schema)
        else:
            # Backup the current required before updating it with the new ones.
            required = set(self._required)
            super().add_schema(schema)
            self._required = required | set(schema["required"])

    def add_object(self, obj: Any) -> None:
        """Add an object and merge the required attribute.

        Args:
            obj: An object to be added.
        """
        if self._required is None:
            super().add_object(obj)
        else:
            # Backup the current required before updating it with the new ones.
            required = set(self._required)
            super().add_object(obj)
            self._required = required | set(obj.keys())


class _MultipleMeta(type(abc.Mapping), type(SchemaBuilder)):
    """Required base class for inheriting from multiple classes with meta classes."""


class MutableMappingSchemaBuilder(abc.Mapping, SchemaBuilder, metaclass=_MultipleMeta):
    """A mutable genson SchemaBuilder with a dictionary-like interface.

    The :class:`SchemaBuilder` does not provide a way to mutate directly the properties
    of a schema (these are stored deeply). For ease of usage, this class brings the
    properties closer to the surface, and the mutability is only provided by the ability
    to delete a property.
    """

    EXTRA_STRATEGIES = (_MergeRequiredStrategy,)

    def __getitem__(self, key: str) -> dict[str, Any]:
        # Immutable workaround because self.properties is a defaultdict.
        if key in self.properties:
            return self.properties[key]
        else:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.properties)

    def __len__(self) -> int:
        return len(self.properties)

    def __delitem__(self, key) -> None:
        del self.properties[key]
        if key in self.required:
            self.required.remove(key)

    @property
    def properties(self) -> dict[str, Any]:
        """Return the properties.

        Returns:
            The existing properties, otherwise an empty dictionary.
        """
        try:
            return self._root_node._active_strategies[0]._properties
        except (AttributeError, IndexError):
            return {}

    @property
    def required(self) -> set[str]:
        """Return the required properties.

        Returns:
            The required properties, otherwise an empty set.
        """
        try:
            required = self._root_node._active_strategies[0]._required
        except (AttributeError, IndexError):
            return set()
        if required is None:
            return set()
        return required
