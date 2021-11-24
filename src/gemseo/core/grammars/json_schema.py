# -*- coding: utf-8 -*-
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

from gemseo.utils.py23_compat import PY2

if PY2:
    from collections import ItemsView, KeysView, Mapping, ValuesView
else:
    from collections.abc import ItemsView, KeysView, Mapping, ValuesView

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

    def add_schema(self, schema):
        """Add a schema and merge the required attribute.

        Args:
            schema: A schema to be added.
        """
        if "required" not in schema or self._required is None:
            super(_MergeRequiredStrategy, self).add_schema(schema)
        else:
            required = set(self._required)
            super(_MergeRequiredStrategy, self).add_schema(schema)
            self._required = required | set(schema["required"])

    def add_object(self, obj):
        """Add an object and merge the required attribute.

        Args:
            obj: An object to be added.
        """
        if self._required is None:
            super(_MergeRequiredStrategy, self).add_object(obj)
        else:
            required = set(self._required)
            super(_MergeRequiredStrategy, self).add_object(obj)
            self._required = required | set(obj.keys())


class MutableMappingSchemaBuilder(SchemaBuilder):
    """A mutable genson SchemaBuilder with a dictionary-like interface.

    The mutability is provided by the ability to delete a property.
    """

    EXTRA_STRATEGIES = (_MergeRequiredStrategy,)

    def get(self, key, default=None):
        """Implement the standard mapping getter.

        Args:
            key: A key whose mapped value shall be returned.
            default: The default value returned if the key is missing.

        Returns:
            The value mapped to key if it exists, default otherwise.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        try:
            self[key]
        except KeyError:
            return False
        else:
            return True

    def keys(self):
        """Return the keys.

        Returns:
            The keys.
        """
        return KeysView(self)

    def items(self):
        """Return the pairs of key and mapped value.

        Returns:
            The pairs of key and mapped value.
        """
        return ItemsView(self)

    def values(self):
        """Return the values.

        Returns:
            The values.
        """
        return ValuesView(self)

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    __reversed__ = None

    @property
    def _properties(self):
        """Return the properties.

        Returns:
            The existing properties, otherwise an empty dictionary.
        """
        try:
            return self._root_node._active_strategies[0]._properties
        except AttributeError:
            return {}

    def __getitem__(self, key):
        return self._properties[key]

    def __iter__(self):
        return iter(self._properties)

    def __len__(self):
        return len(self._properties)

    def __delitem__(self, key):
        del self._properties[key]
        required_properties = self._root_node._active_strategies[0]._required
        if required_properties is None:
            return
        try:
            required_properties.remove(key)
        except KeyError:
            pass

    # For backward compatibility
    # TODO: add deprecation warning
    to_dict = SchemaBuilder.to_schema
