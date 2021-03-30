# -*- coding: utf-8 -*-
# The MIT License (MIT)
#
# Copyright (c) 2014 Jon Wolverton github.com/wolverdude
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
A JSON schema generation tool
*****************************
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from builtins import dict, str, super
from collections import defaultdict
import json
import re

from future import standard_library
from six import string_types


standard_library.install_aliases()


JS_TYPES = {dict: 'object',
            list: 'array',
            str: 'string',
            type('a'): 'string',
            type(u'a'): 'string',
            int: 'integer',
            float: 'number',
            bool: 'boolean',
            type(None): 'null'}

for typ in string_types:
    JS_TYPES[typ] = "string"


class SchemaDefaultDict(defaultdict):
    """A defaultdict implementation that is serializable with pickle
    and that returns an empty Schema by default
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor
        """
        super(SchemaDefaultDict, self).__init__()
        self.options = {}

    def __missing__(self, key):
        value = Schema(**self.options)
        self[key] = value
        return value


class Schema(object):
    """Basic schema generator class. Schema objects can be loaded up
    with existing schemas and objects before being serialized.
    """

    def __init__(self, merge_arrays=True, additional_items=True,
                 additional_props=True, match_props=[],
                 exclude_at_merge=None):
        """Builds a schema generator object.

        * `merge_arrays` (default `True`): Assume all array items share
          the same schema (as they should). The alternate behavior is to
          create a different schema for each item in every array.
        * `additional_items` (default 'True'): If True, allow
          tuple-validated arrays to be followed by unvalidated items.
          If False, generate an "additionalItems": False element so that
          array items not specified in the schema cause a ValidationError.
          Not used with list-validated (merged) arrays.
        * `additional_props` (default 'True'): If True, allow objects
          to include unvalidated properties.  If False, generate an
          "additionalProperties": False element so that properties not
          specified in the schema cause a ValidationError.
        * `match_props` (default '[]'): List of regular expressions to
          compare with property keys.  Properties with matching
          keys share the same "patternProperties" schema.

        """

        self._options = {
            'merge_arrays': merge_arrays,
            'additional_items': additional_items,
            'additional_props': additional_props,
            'match_props': match_props,
        }
        self._type = set()
        self._description = None
        self._required = None
        self._exclude_at_merge = exclude_at_merge
        if self._exclude_at_merge is None:
            self._exclude_at_merge = []
# Original implementation
#         self._properties = DictWithDefaultSchema(**self._options)
#         self._patternProperties = DictWithDefaultSchema(**self._options)
        self._patternProperties = SchemaDefaultDict()
        self._patternProperties.options = self._options
        self._properties = SchemaDefaultDict()
        self._properties.options = self._options
        self._items = []
        self._other = {}

    def add_schema(self, schema):
        """Merges in an existing schema.

        :param schema: required
        :param an: existing JSON Schema to merge
        """

        # serialize instances of Schema before parsing
        if isinstance(schema, Schema):
            schema = schema.to_dict()

        # parse properties and add them individually
        for prop, val in list(schema.items()):
            if prop == 'type':
                self._add_type(val)
            elif prop == 'required':
                self._add_required(val)
            elif prop in ['properties', 'patternProperties']:
                self._add_properties(prop, val, 'add_schema')
            elif prop == 'items':
                self._add_items(val, 'add_schema')
            elif prop == "description":
                self.add_description(val)
            elif prop not in self._other:
                self._other[prop] = val
            elif self._other[prop] != val and prop not in\
                    self._exclude_at_merge:
                e = prop + ': ' + str(self._other[prop]) + ' ^= ' + str(val)
                raise SchemaError('schema incompatible -- ' + e)

        # make sure the 'required' key gets set regardless
        if 'required' not in schema:
            self._add_required([])

        # return self for easy method chaining
        return self

    def add_object(self, obj, descr=None):
        """Modify the schema to accomodate an object.

        :param obj: required
        :param a: JSON object to use in generate the schema
        """

        if isinstance(obj, dict):
            self._generate_object(obj, descr)
        elif isinstance(obj, list):
            self._generate_array(obj, descr)
        else:
            self._generate_basic(obj, descr)

        # return self for easy method chaining
        return self

    def to_dict(self):
        """Convert the current schema to a `dict`.

        """
        # start with existing fields
        schema = dict(self._other)

        if 'additionalItems' in schema:
            if schema['additionalItems'] == True or not isinstance(
                    self._items, list):
                del(schema['additionalItems'])
        if 'additionalProperties' in schema and schema[
                'additionalProperties'] == True:
            del(schema['additionalProperties'])

        # unpack the type field
        if self._type:
            schema['type'] = self._get_type()

        if self._description:
            schema["description"] = self._get_description()

        # call recursively on subschemas if object or array
        if 'object' in self._type:
            props = self._get_properties()
            # include unnecessary but valid "properties": {}
            if props[0] or not props[1]:
                schema['properties'] = props[0]
            if props[1]:
                schema['patternProperties'] = props[1]
            if self._required:
                schema['required'] = self._get_required()
        if 'array' in self._type:
            items = self._get_items()
            if items or isinstance(items, dict):
                schema['items'] = items
        return schema

    def to_json(self, *args, **kwargs):
        """Convert the current schema directly to serialized JSON.

        :param args:
        :param kwargs:
        """
        return json.dumps(self.to_dict(), *args, **kwargs)

    # private methods

    # getters

    def _get_type(self):
        """ """
        schema_type = self._type | set()  # get a copy

        # remove any redundant integer type
        if 'integer' in schema_type and 'number' in schema_type:
            schema_type.remove('integer')

        # unwrap if only one item, else convert to array
        if len(schema_type) == 1:
            (schema_type,) = schema_type
        else:
            schema_type = sorted(schema_type)

        return schema_type

    def _get_description(self):
        return self._description

    def _get_required(self):
        """ """
        return sorted(self._required) if self._required else []

    def _get_properties(self):
        """


        """
        properties, patprops = {}, {}
        for prop, subschema in list(self._properties.items()):
            properties[prop] = subschema.to_dict()
        for prop, subschema in list(self._patternProperties.items()):
            patprops[prop] = subschema.to_dict()
        return ((properties, patprops))

    def _get_items(self):
        """
        Lists items

        """
        if isinstance(self._items, list):
            return [subschema.to_dict() for subschema in self._items]
        else:
            return self._items.to_dict()

    # setters

    def _add_type(self, val_type):
        """

        :param val_type:
        """
        if isinstance(val_type, string_types):
            self._type.add(val_type)
        else:
            self._type |= set(val_type)

    def _add_required(self, required):
        """

        :param required:

        """
        if self._required is None:
            # if not already set, set to this
            self._required = set(required)
        else:
            # use intersection to limit to properties present in both
            self._required.update(set(required))

    def _add_properties(self, ptype, properties, func):
        """

        :param ptype: param properties:
        :param func:
        :param properties:
        """
        # recursively modify subschemas
        pattern = self._options['match_props']
        if pattern and not ptype:
            self._add_properties_merge(pattern, properties, func)
        else:
            self._add_properties_sep(ptype or 'properties', properties, func)

    def _add_properties_merge(self, pattern, properties, func):
        """

        :param pattern: param properties:
        :param func:
        :param properties:
        """
        err = None
        for prop, val in list(properties.items()):
            match = []
            for pat in pattern:
                m = re.search(pat, prop)
                if m:
                    getattr(self._patternProperties[pat], func)(val)
                    match.append(pat)
                    self._required -= set((prop,))
            if len(match) > 1:
                err = (prop, match)
            elif len(match) < 1:
                getattr(self._properties[prop], func)(val)
        if err:
            raise SchemaError('patternProperties multiple match: ' +
                              err[0] + ' ~= ' + ', '.join(err[1]))

    def _add_properties_sep(self, ptype, properties, func):
        """

        :param ptype: param properties:
        :param func:
        :param properties:
        """
        pdict = self._properties if ptype == 'properties' else self._patternProperties
        for prop, val in list(properties.items()):
            getattr(pdict[prop], func)(val)

    def _add_items(self, items, func):
        """

        :param items: param func:
        :param func:
        """
        if self._options['merge_arrays']:
            self._add_items_merge(items, func)
        else:
            self._add_items_sep(items, func)

    def add_description(self, descr):
        """

        :param descr: tue description
        """
        self._description = descr

    def _add_items_merge(self, items, func):
        """

        :param items: param func:
        :param func:
        """
        if not self._items:
            self._items = Schema(**self._options)
        method = getattr(self._items, func)
        if isinstance(items, list):
            for item in items:
                method(item)
        else:
            method(items)

    def _add_items_sep(self, items, func):
        """

        :param items: param func:
        :param func:
        """
        for item in items:
            subschema = Schema(**self._options)
            getattr(subschema, func)(item)
            self._items.append(subschema)

    def _add_additionalItems(self):
        """ """
        self._other['additionalItems'] = self._options['additional_items']

    def _add_additionalProperties(self):
        """ """
        self._other['additionalProperties'] = self._options['additional_props']

    # generate from object

    def _generate_object(self, obj, descr=None):
        """

        :param obj:
        """
        self._add_type('object')
        self._add_required(list(obj.keys()))
        self._add_properties(None, obj, 'add_object')
        self._add_additionalProperties()
        if descr is not None:
            self._add_properties(None, descr, 'add_description')

    def _generate_array(self, array, descr=None):
        """

        :param array:
        """
        self._add_type('array')
        self._add_items(array, 'add_object')
        self._add_additionalItems()
        if descr is not None:
            self.add_description(descr)

    def _generate_basic(self, val, descr=None):
        """

        :param val:
        """
        val_type = JS_TYPES[type(val)]
        if val_type != "null":
            self._add_type(val_type)
        if descr is not None:
            self.add_description(descr)


class SchemaError(Exception):
    """ """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
