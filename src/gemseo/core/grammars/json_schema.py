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

from abc import ABCMeta
from collections.abc import MutableMapping
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import TypedDict
from typing import cast

from genson import SchemaBuilder
from genson import SchemaNode
from genson.schema.builder import _MetaSchemaBuilder
from genson.schema.strategies import Number
from genson.schema.strategies import Object
from numpy import float64
from numpy import int64

from gemseo.core.grammars._utils import NOT_IN_THE_GRAMMAR_MESSAGE
from gemseo.typing import StrKeyMapping

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import NotRequired

    Property = TypedDict(  # noqa: UP013
        "Property",
        {"type": str, "items": NotRequired["Property"]},
        total=False,
    )

    Properties = dict[str, Property]

    Schema = TypedDict(  # noqa: UP013
        "Schema",
        {
            "properties": Properties,
            "type": str,
            "id": NotRequired[str],
            "required": NotRequired[list[str]],
            "$schema": str,
        },
        total=False,
    )

    Obj = StrKeyMapping

SchemaBuilderProperties = MutableMapping[str, SchemaNode]


class _MergeStrategy(Object):  # type: ignore[misc]
    """A genson strategy to either merge or update a schema.

    By default, genson merges nodes, the update is triggered via a class attribute.
    """

    # Do not merge the name and id properties.
    KEYWORDS = (*Object.KEYWORDS, "name", "id")

    update: ClassVar[bool] = False
    """Whether to update or merge the schema."""

    @contextmanager
    def __handle_update(self) -> Iterator[None]:
        """A context manager to handle the update vs merge."""
        # Pass the update switch to _SchemaNode.
        self.node_class.update = self.update
        yield
        # Reset to the merge behavior because _SchemaNode may be used by other instances
        # that should merge.
        self.node_class.update = False

    def add_schema(self, schema: StrKeyMapping) -> None:
        with self.__handle_update():
            super().add_schema(schema)

    def add_object(self, obj: StrKeyMapping) -> None:
        with self.__handle_update():
            super().add_object(obj)


class _SchemaNode(SchemaNode):  # type: ignore[misc]
    """Overload :meth:`.add_schema` and :meth:`.add_object` to allow updating.

    By default, genson merges nodes, the update is triggered via a class attribute.
    """

    update: ClassVar[bool] = False
    """Whether to update or merge the schema."""

    def add_schema(self, schema: StrKeyMapping) -> None:
        self.__handle_update()
        super().add_schema(schema)

    def add_object(self, obj: Obj) -> None:
        self.__handle_update()
        super().add_object(obj)

    def __handle_update(self) -> None:
        """Handle the update or merge behavior.

        When updating, the already existing active strategies are removed such that only
        the last one added remains.
        """
        if self.update and self._active_strategies:
            self._active_strategies.clear()


class _MultipleMeta(ABCMeta, _MetaSchemaBuilder):  # type: ignore[misc]
    """Required metaclass for inheriting from multiple classes with metaclasses.

    Also fix the ``NODE_CLASS`` overloading because it does not use the ``NODE_CLASS``
    passed to a class derived from ``SchemaBuilder``.
    """

    def __init__(cls, name: str, bases: tuple[type], attrs: dict[str, Any]) -> None:
        super().__init__(name, bases, attrs)
        cls.NODE_CLASS = type(
            f"{name}SchemaNode", (_SchemaNode,), {"STRATEGIES": cls.STRATEGIES}
        )


class _Number(Number):  # type: ignore[misc]
    """A number strategy that handles numpy data."""

    PYTHON_TYPES = (*Number.PYTHON_TYPES, float64, int64)


class MutableMappingSchemaBuilder(
    StrKeyMapping,
    SchemaBuilder,  # type: ignore[misc]
    metaclass=_MultipleMeta,
):
    """A mutable genson SchemaBuilder with a dictionary-like interface.

    The :class:`SchemaBuilder` does not provide a way to mutate directly the properties
    of a schema (these are stored deeply). For ease of usage, this class brings the
    properties closer to the surface, and the mutability is only provided by the ability
    to delete a property.
    """

    EXTRA_STRATEGIES = (_MergeStrategy, _Number)
    NODE_CLASS = _SchemaNode

    def __getitem__(self, key: str) -> SchemaNode:
        self.check_property_names(key)
        return self.properties[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.properties)

    def __len__(self) -> int:
        return len(self.properties)

    def __delitem__(self, key: str) -> None:
        del self.properties[key]

    @property
    def properties(self) -> SchemaBuilderProperties:
        """Return the properties.

        Returns:
            The existing properties, otherwise an empty dictionary.
        """
        try:
            return cast(
                "SchemaBuilderProperties",
                self._root_node._active_strategies[0]._properties,
            )
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
        return cast("set[str]", required)

    def check_property_names(self, *names: str) -> None:
        """Check that the names are existing properties.

        Args:
            *names: The names to be checked.

        Raises:
            KeyError: If a name is not an existing property.
        """
        for name in names:
            if name not in self.properties:
                msg = NOT_IN_THE_GRAMMAR_MESSAGE.format(name)
                raise KeyError(msg)

    def add_schema(self, schema: Schema, update: bool) -> None:
        """
        Args:
            update: Whether to update or merge the schema.
        """  # noqa: D205 D212 D415
        _MergeStrategy.update = update
        super().add_schema(schema)

    def add_object(self, obj: Obj, update: bool) -> None:
        """
        Args:
            update: Whether to update or merge the schema.
        """  # noqa: D205 D212 D415
        _MergeStrategy.update = update
        super().add_object(obj)
