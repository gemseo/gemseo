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
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from genson import SchemaBuilder
from genson import SchemaNode
from genson.schema.builder import _MetaSchemaBuilder
from genson.schema.strategies import Number
from genson.schema.strategies import Object
from numpy import float64
from numpy import int64

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping


class _MergeRequiredStrategy(Object):
    """A genson Object with a modified required attribute handling.

    Genson Object does not merge the required attribute on purpose.
    See :ref:`https://github.com/wolverdude/GenSON#genson`.
    This class will merge the required attributes.

    By default, genson merges nodes, the update is triggered via a class attribute.
    """

    # Do not merge the name and id properties.
    KEYWORDS = (*Object.KEYWORDS, "name", "id")

    update: ClassVar[bool] = False
    """Whether to update or merge the schema."""

    @contextmanager
    def __handle_update(
        self, updated_names: Iterable[str], required: Iterable[str]
    ) -> None:
        """A context manager to handle the update vs merge.

        Args:
            updated_names: All the names to update.
            required: The required names to add.
        """
        # Pass the update switch to _SchemaNode.
        self.node_class.update = self.update

        if not self._required or not required:
            yield
        else:
            # Backup the current required before updating it with the new ones.
            _required = set(self._required)
            yield
            if self.update:
                # The elements we update from overrule the existing ones, the required
                # or not state for all the elements we update shall be reset.
                _required -= updated_names
            self._required = _required | set(required)

        # Reset to the merge behavior because _SchemaNode may be used by other instances
        # that should merge.
        self.node_class.update = False

    def add_schema(self, schema: Mapping[str, Any]) -> None:
        """Add a schema and merge the required attribute.

        Args:
            schema: A schema to be added.
        """
        with self.__handle_update(
            schema.get("properties", {}).keys(), schema.get("required")
        ):
            super().add_schema(schema)

    def add_object(self, obj: Mapping[str, Any]) -> None:
        """Add an object and merge the required attribute.

        Args:
            obj: An object to be added.
        """
        with self.__handle_update(obj.keys(), obj.keys()):
            super().add_object(obj)


class _SchemaNode(SchemaNode):
    """Overload :meth:`.add_schema` and :meth:`.add_object` to allow updating.

    By default, genson merges nodes, the update is triggered via a class attribute.
    """

    update: ClassVar[bool] = False
    """Whether to update or merge the schema."""

    def add_schema(self, schema) -> None:
        self.__handle_update()
        super().add_schema(schema)

    def add_object(self, obj) -> None:
        self.__handle_update()
        super().add_object(obj)

    def __handle_update(self) -> None:
        """Handle the update or merge behavior.

        When updating, the already existing active strategies are removed such that only
        the last one added remains.
        """
        if self.update and self._active_strategies:
            self._active_strategies.clear()


class _MultipleMeta(type(abc.Mapping), _MetaSchemaBuilder):
    """Required meta class for inheriting from multiple classes with meta classes.

    Also fix the ``NODE_CLASS`` overloading because it does not use the ``NODE_CLASS``
    passed to a class derived from ``SchemaBuilder``.
    """

    def __init__(cls, name: str, bases: tuple(type), attrs: dict[str, Any]) -> None:
        super().__init__(name, bases, attrs)
        cls.NODE_CLASS = type(
            "%sSchemaNode" % name, (_SchemaNode,), {"STRATEGIES": cls.STRATEGIES}
        )


class _Number(Number):
    """A number strategy that handles numpy data."""

    PYTHON_TYPES = (*Number.PYTHON_TYPES, float64, int64)


class MutableMappingSchemaBuilder(abc.Mapping, SchemaBuilder, metaclass=_MultipleMeta):
    """A mutable genson SchemaBuilder with a dictionary-like interface.

    The :class:`SchemaBuilder` does not provide a way to mutate directly the properties
    of a schema (these are stored deeply). For ease of usage, this class brings the
    properties closer to the surface, and the mutability is only provided by the ability
    to delete a property.
    """

    EXTRA_STRATEGIES = (_MergeRequiredStrategy, _Number)
    NODE_CLASS = _SchemaNode

    def __getitem__(self, key: str) -> dict[str, Any]:
        self.check_property_names(key)
        return self.properties[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.properties)

    def __len__(self) -> int:
        return len(self.properties)

    def __delitem__(self, key: str) -> None:
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

    def check_property_names(self, *names: str) -> None:
        """Check that the names are existing properties.

        Args:
            *names: The names to be checked.

        Raises:
            KeyError: If a name is not an existing property.
        """
        for name in names:
            if name not in self.properties:
                raise KeyError(f"The name {name} is not in the grammar.")

    def add_schema(self, schema, update: bool) -> None:
        """
        Args:
            update: Whether to update or merge the schema.
        """  # noqa: D205 D212 D415
        _MergeRequiredStrategy.update = update
        super().add_schema(schema)

    def add_object(self, obj, update: bool) -> None:
        """
        Args:
            update: Whether to update or merge the schema.
        """  # noqa: D205 D212 D415
        _MergeRequiredStrategy.update = update
        super().add_object(obj)
