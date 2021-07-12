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

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""A Grammar based on JSON schema."""

from __future__ import division, unicode_literals

import json
import logging
from numbers import Number

from numpy import generic, ndarray, zeros

from gemseo.core.grammars.abstract_grammar import AbstractGrammar
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.json_schema import MutableMappingSchemaBuilder
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.utils.py23_compat import PY2, JsonSchemaException, Path, compile_schema

LOGGER = logging.getLogger(__name__)


class JSONGrammar(AbstractGrammar):
    """A grammar subclass that stores the input or output data types and structure a
    MDODiscipline using a JSON format It is able to check the inputs and outputs against
    a JSON schema."""

    PROPERTIES_FIELD = "properties"
    REQUIRED_FIELD = "required"
    TYPE_FIELD = "type"
    OBJECT_FIELD = "object"
    TYPES_MAP = {
        "array": ndarray,
        "float": float,
        "string": str,
        "integer": int,
        "boolean": bool,
        "number": Number,
    }

    def __init__(
        self,
        name,
        schema_file=None,
        schema=None,
        grammar_type=None,
        descriptions=None,
    ):
        """Constructor.

        :param name : grammar name
        :param schema : a genson Schema  to initialize self
        :param schema_file : the json schema file
        """
        super(JSONGrammar, self).__init__(name)
        self._validator = None
        self.schema = None
        self._init_schema()

        if schema is not None:
            self.schema.add_schema(schema)
        elif schema_file is not None:
            self.init_from_schema_file(schema_file, descriptions=descriptions)
        else:
            self.initialize_from_base_dict({}, description_dict=descriptions)

    def __repr__(self):
        return "{}, schema: {}".format(self, self.schema.to_json())

    def _init_schema(self):
        """Initialize the schema."""
        self.schema = MutableMappingSchemaBuilder()

    def clear(self):
        """Clears the data to produce an empty grammar."""
        self.__set_grammar_from_dict({})

    def _init_validator(self):
        """Initialize the validator."""
        schema_dict = self.schema.to_schema()
        schema_dict.pop("id", None)
        self._validator = compile_schema(schema_dict)

    @classmethod
    def cast_array_to_list(cls, data_dict):
        """Casts the numpy arrays to lists for dictionary values.

        :param data_dict: the data dictionary
        :returns: The dict with casted arrays
        """
        out_d = data_dict.copy()
        for key, value in data_dict.items():
            if isinstance(value, (ndarray, generic)):
                out_d[key] = value.real.tolist()
            elif isinstance(value, dict):
                out_d[key] = cls.cast_array_to_list(value)
        return out_d

    def load_data(self, data, raise_exception=True):
        """Loads the data dictionary in the grammar and checks it against json schema.

        :param data: the input data
        :param raise_exception: if False, no exception is raised
            when data is invalid (Default value = True)
        """
        if not isinstance(data, dict):
            raise InvalidDataException(
                "Input data must be a dictionary: got a {} instead".format(type(data))
            )

        if self._validator is None:
            self._init_validator()

        error_exist = False
        data_to_check = self.cast_array_to_list(data)

        try:
            self._validator(data_to_check)
        except JsonSchemaException as error:
            error_exist = True
            msg = "Invalid data in: {}".format(self.name)

            if error.args[0].startswith("data must contain"):
                # Error messages are not clear enough when missing properties
                # All keys are put in the message
                diff = sorted(set(self.get_data_names()) - set(data.keys()))
                if diff:
                    msg += "\nMissing mandatory properties: {}".format(",".join(diff))
                else:
                    msg += "\n', error: {}".format(error.args[0])
            else:
                msg += "\n', error: {}".format(error.args[0])

            LOGGER.error(msg)

        if error_exist and raise_exception:
            err = "Invalid data from grammar {}".format(self.name)
            raise InvalidDataException(err)

        # Check a copy to keep types and arrays but store initial dict for complex
        # Add defaults
        for key, value in data_to_check.items():
            data.setdefault(key, value)

        return data

    def init_from_schema_file(self, schema_path, descriptions=None):
        """Set the grammar from a file.

        :param schema_path: path to the schema file.
        """
        schema_path = Path(schema_path)

        if not schema_path.exists():
            msg = "Try to initialize grammar with not existing file: {}".format(
                schema_path
            )
            raise FileNotFoundError(msg)

        schema_dict = json.loads(schema_path.read_text())
        self.__set_grammar_from_dict(schema_dict, descriptions)

    def __set_grammar_from_dict(self, schema_dict, descriptions=None):
        """Set the grammar from a dictionary.

        Args:
            schema_dict: Schema dictionary.
            descriptions: Properties descriptions, optional.
        """
        self._init_schema()
        self.__update_grammar_from_dict(schema_dict, descriptions)

    def __update_grammar_from_dict(self, schema_dict, descriptions=None):
        """Update the grammar from a dictionary.

        Args:
            schema_dict: Schema dictionary.
            descriptions: Properties descriptions, optional.
        """
        if descriptions is not None:
            if not isinstance(schema_dict, dict):
                schema_dict = schema_dict.to_schema()
            for ppty_name, ppty_schema in schema_dict["properties"].items():
                try:
                    ppty_schema["description"] = descriptions[ppty_name]
                except KeyError:
                    LOGGER.debug(
                        "skipping description for unknown property %s", ppty_name
                    )

        self.__merge_schema(schema_dict)

    def __merge_schema(self, schema):
        """Merge a schema in the current one.

        Args:
            schema: Schema to merge, could be a schema object or a dictionary.
        """
        self.schema.add_schema(schema)
        self._validator = None

    def initialize_from_data_names(
        self,
        data_names,
        descriptions=None,
    ):
        """Initializes a JSONGrammar from a list of data. All data of the grammar will
        be set as arrays.

        :param data_names: a data names list
        """
        data = zeros(1)
        typical_data_dict = {k: data for k in data_names}
        self.initialize_from_base_dict(typical_data_dict, description_dict=descriptions)

    def initialize_from_base_dict(
        self,
        typical_data_dict,
        description_dict=None,
    ):
        """Initialize a json grammar with types and names from a typical data entry. The
        keys of the typical_data_dict will be the names of the data in the grammar. The
        types of the values of the typical_data_dict will be converted to JSON Schema
        types and define the properties of the JSON Schema.

        :param typical_data_dict: a data dictionary with keys as data names
            and values as a typical value for this data
        :param description_dict: dictionary of descriptions,
             {name:meaning} structure
        """
        # Convert arrays to list as for check
        list_data_dict = self.cast_array_to_list(typical_data_dict)
        self.schema.add_object(list_data_dict)
        self.__set_grammar_from_dict(self.schema, description_dict)

    def get_data_names(self):
        """Returns the list of data names.

        :returns: the data names, as a dict keys set
        """
        return list(self.schema.keys())

    def is_data_name_existing(self, data_name):
        """Checks if data_name is present in grammar.

        :param data_name: the data name
        :returns: True if data is in grammar
        """
        return data_name in self.get_data_names()

    def is_all_data_names_existing(self, data_names):
        """Checks if data_names are present in grammar.

        :param data_names: the data names list
        :returns: True if all data are in grammar
        """
        exists = self.is_data_name_existing
        for data_name in data_names:
            if not exists(data_name):
                return False
        return True

    def update_from(self, input_grammar):
        """Adds properties coming from another grammar.

        :param input_grammar: the grammar to take inputs from
        """
        if not isinstance(input_grammar, JSONGrammar):
            msg = (
                "A {} is expected as input, but an object of type {} "
                "has been provided.".format(self.__class__, type(input_grammar))
            )
            raise TypeError(msg)

        self.__merge_schema(input_grammar.schema)

    def to_simple_grammar(self):
        """Creates a SimpleGrammar from self, preserving the features if possible
        Ignores the features of JSONGrammar that are not supported by SimpleGrammar.

        :returns: a SimpleGrammar instance
        """
        grammar = SimpleGrammar(self.name)
        schema_dict = self.schema.to_schema()
        properties = schema_dict.get(self.PROPERTIES_FIELD, {})

        for prop, desc in properties.items():
            grammar.data_names.append(prop)
            if "type" not in desc or desc["type"] not in self.TYPES_MAP:
                d_type = None
            else:
                d_type = self.TYPES_MAP[desc["type"]]
            grammar.data_types.append(d_type)

            if d_type == "array":
                sub_type = desc["items"].get("type")
                if sub_type not in ["number", "integer", None]:
                    msg = "Unsupported type {sub_type} in JSONGrammar {name}"
                    msg += "for property {prop} in conversion to simple grammar"
                    err = msg.format(sub_type=sub_type, name=self.name, prop=prop)
                    LOGGER.warning(err)

            for prop_name in ["minItems", "maxItems", "additionalItems", "contains"]:
                if prop_name in desc:
                    msg = "Unsupported feature {desc} in JSONGrammar {name}"
                    msg += "for property {prop} in conversion to simple grammar"
                    err = msg.format(desc=desc, name=self.name, prop=prop)
                    LOGGER.warning(err)

        return grammar

    def update_from_if_not_in(self, input_grammar, exclude_grammar):
        """Adds objects coming from input_grammar if they are not in exclude_grammar.

        :param input_grammar: the grammar to take inputs from
        :param exclude_grammar: exclusion grammar
        """
        if not (
            isinstance(input_grammar, self.__class__)
            and isinstance(exclude_grammar, self.__class__)
        ):
            msg = self._get_update_error_msg(self, input_grammar, exclude_grammar)
            raise TypeError(msg)

        schema = MutableMappingSchemaBuilder()
        schema.add_schema(input_grammar.schema)

        for name in exclude_grammar.schema.keys():
            try:
                del schema[name]
            except KeyError:
                pass

        self.__merge_schema(schema)

    def restrict_to(self, data_names):
        """Restrict the grammar to given names.

        :param data_names: the names of the data to restrict the grammar to
        """
        for name in list(self.schema.keys()):
            if name not in data_names:
                self.remove_item(name)

    def remove_item(self, item_name):
        """Removes an item from the grammar.

        :param item_name: the item name to be removed
        """
        del self.schema[item_name]

    def set_item_value(self, item_name, item_value):
        """Sets the value of an item.

        :param item_name: the item name to be modified
        :param item_value: value of the item
        """
        if not self.is_data_name_existing(item_name):
            raise ValueError("Item {} not in grammar {}".format(item_name, self.name))
        self_dict = self.schema.to_schema()
        self_dict[self.PROPERTIES_FIELD][item_name] = item_value

        self.__set_grammar_from_dict(self_dict)

    def write_schema(self, path=None):
        """Write the schema to a file.

        Args:
            path: Write to this path, if None then write to a file named after the
                grammar and with .json extension.
        """
        if path is None:
            path = Path(self.name).with_suffix(".json")
        else:
            path = Path(path)

        schema_json = self.schema.to_json()

        if PY2:
            # workaround, see https://stackoverflow.com/a/36003774
            x = json.dumps(
                schema_json,
                ensure_ascii=False,
            )
            if isinstance(x, str):
                x = unicode(x, "UTF-8")  # noqa: F821
            path.write_text(x)
        else:
            json.dump(
                schema_json,
                path.open("w", encoding="utf-8"),
            )

    def __getstate__(self):
        """Used by pickle to define what to serialize.

        :returns: the dict to serialize
        """
        out_d = dict(self.__dict__)
        out_d.pop("_validator")
        # genson schema cannot be pickled: use its dictionary representation
        out_d["schema"] = self.schema.to_schema()
        return out_d

    def __setstate__(self, data_dict):
        """Used by pickle to define what to deserialize.

        :param data_dict : update self dict from data_dict to deserialize
        """
        self.__dict__.update(data_dict)
        # genson schema cannot be pickled: use its dictionary representation
        self.__set_grammar_from_dict(data_dict.pop("schema"))
