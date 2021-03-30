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
"""
A Grammar based on JSON schema
******************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

from future import standard_library
from numpy import generic, ndarray, zeros

from gemseo import LOGGER
from gemseo.core.grammar import AbstractGrammar, InvalidDataException
from gemseo.third_party.genson_generator import Schema
from gemseo.utils.py23_compat import JsonSchemaException, compile_schema

standard_library.install_aliases()


class JSONGrammar(AbstractGrammar):
    """A grammar subclass that stores the input or output data types
    and structure a MDODiscipline using a JSON format
    It is able to check the inputs and outputs against a JSON schema.
    """

    PROPERTIES_FIELD = "properties"
    DEFAULTS_FIELD = "defaults"
    REQUIRED_FIELD = "required"
    TYPE_FIELD = "type"
    OBJECT_FIELD = "object"

    def __init__(
        self,
        name,
        schema_file=None,
        schema=None,
        grammar_type=AbstractGrammar.INPUT_GRAMMAR,
    ):
        """
        Constructor
        :param name : grammar name
        :param schema : a genson Schema  to initialize self
        :param schema_file : the json schema file
        :param grammar_type : the type of grammar : input or output
        """
        super(JSONGrammar, self).__init__()
        self.schema = None
        self._properties = None
        self._validator = None
        self.data = None
        self.name = name
        self.grammar_type = grammar_type
        self._init_schema()
        if schema is not None:
            self.schema.add_schema(schema)
            self._update_properties()
        if schema_file is not None:
            self.init_from_schema_file(schema_file)
        if schema is None and schema_file is None:
            self.initialize_from_base_dict(typical_data_dict={})

    def _init_schema(self):
        """Initializes the genson schema"""
        self.schema = Schema(merge_arrays=True, exclude_at_merge=["name", "id"])

    def _update_properties(self):
        """Updates the properties after a change in the schema"""
        schema_dict = self.schema.to_dict()
        self._properties = schema_dict.get(self.PROPERTIES_FIELD)
        if self._properties is None:
            self._properties = {}

        self._validator = None

    @property
    def properties(self):
        """Accessor for the properties of the schema"""
        return self._properties

    def clear(self):
        """Clears the data to produce an empty grammar"""
        self._init_schema()
        self._update_properties()
        self._validator = None

    def _init_validator(self):
        """Initializes the validator according to self"""
        schema_dict = self.schema.to_dict()
        schema_dict.pop("id", None)
        self._validator = compile_schema(schema_dict)

    @staticmethod
    def cast_array_to_list(data_dict):
        """
        Casts the numpy arrays in data_dict to lists

        :param data_dict : the data dictionary
        :returns: The dict with casted arrays
        """
        out_d = data_dict.copy()
        for key, value in data_dict.items():
            if isinstance(value, (ndarray, generic)):
                out_d[key] = value.real.tolist()
            elif isinstance(value, dict):
                out_d[key] = JSONGrammar.cast_array_to_list(value)
        return out_d

    def load_data(self, data_dict, raise_exception=True):
        """Loads the data dictionary in the grammar
        and checks it against json schema

        :param data_dict: the input data
        :param raise_exception: if False, no exception is raised
            when data is invalid (Default value = True)
        """
        if not isinstance(data_dict, dict):
            raise InvalidDataException(
                "Input data must be a python dictionary, got "
                + str(type(data_dict))
                + " instead."
            )
        error_exist = False
        data_to_check = self.cast_array_to_list(data_dict)

        if self._validator is None:
            self._init_validator()

        try:
            self._validator(data_to_check)
        except JsonSchemaException as error:
            error_exist = True
            msg = "Invalid data in : " + str(self.name)

            if error.args[0].startswith("data must contain"):
                # Error messages are not clear enough when missing properties
                # All keys are put in the message
                diff = set(set(self.get_data_names()) - set(data_dict.keys()))
                if len(diff) > 0:
                    msg += "\nMissing mandatory properties: " + str(list(diff))
                else:
                    msg += "\n', error : " + str(error.args[0])
            else:
                msg += "\n', error : " + str(error.args[0])
            LOGGER.error(msg)

        if error_exist and raise_exception:
            raise InvalidDataException("Invalid data from grammar " + str(self.name))

        # Check a copy to keep types and arrays but store initial dict for
        # complex
        self.data = data_dict
        # Add defaults
        for key, value in data_to_check.items():
            k_utf = key
            self.data.setdefault(k_utf, value)
        return self.data

    def init_from_schema_file(self, schema_file="input.json"):
        """Initializes grammar from

        :param schema_file: path to the schema input file
            (Default value = "input.json")
        """
        if not os.path.exists(schema_file):
            raise ValueError(
                "Try to initialize grammar with not existing file : " + str(schema_file)
            )

        # Refs are not supported any more for simplification
        # curr_folder = os.path.dirname(schema_file)
        # registry = providers.FilesystemProvider(curr_folder)

        with open(schema_file, "r") as in_file:
            json_content = json.loads(in_file.read())
            # Refs not supported any more for simplification
            # Resolve #ref tags in json
            #             try:
            #              resolved_json = resolve(json_content, '#/properties', registry)
            #             except ValueError as err_json:
            #                 LOGGER.error(err_json)
            #                 raise Exception(
            #                     "Cannot resolve referenced json properties of file : " +
            #                     str(schema_file))
            self.schema.add_schema(json_content)
            self._update_properties()

    def __str__(self):
        msg = "Grammar named :" + str(self.name)
        msg += ", schema = " + self.schema.to_json()
        return msg

    def initialize_from_data_names(
        self, data_names, schema_file=None, write_schema=False
    ):
        """Initializes a JSONGrammar from a list of data.
        All data of the grammar will be set as arrays

        :param data_names: a data names list
        :param schema_file: the output json file path. If None : input.json or
            output.json depending on grammar type.
            (Default value = None)
        :param write_schema: if True, writes the schema files
            (Default value = False)
        """
        data = zeros(1)
        typical_data_dict = {k: data for k in data_names}
        self.initialize_from_base_dict(typical_data_dict, schema_file, write_schema)

    def initialize_from_base_dict(
        self,
        typical_data_dict,
        schema_file=None,
        write_schema=False,
        description_dict=None,
    ):
        """Initialize a json grammar with types and names from a
        typical data entry.
        The keys of the typical_data_dict will be the names of the
        data in the grammar.
        The types of the values of the typical_data_dict will be converted
        to JSON Schema types and define the properties of the JSON Schema.

        :param typical_data_dict: a data dictionary with keys as data names
            and values as a typical value for this data
        :param schema_file: the output json file path. If None : input.json or
            output.json depending on grammar type.
            (Default value = None)
        :param write_schema: if True, writes the schema files
            (Default value = False)
        :param description_dict: dictionary of descriptions,
             {name:meaning} structure
        """
        # Convert arrays to list as for check
        list_data_dict = self.cast_array_to_list(typical_data_dict)
        #         if PY2:
        #             list_data_dict = self.cast_str_val_ascii(list_data_dict)
        self.schema.add_object(list_data_dict, description_dict)

        if write_schema:
            if schema_file is None:
                schema_file = self.name + ".json"
            with open(schema_file, "w") as outf:
                outf.write(self.schema.to_json())
        self._update_properties()

    #     @staticmethod
    #     def cast_str_val_ascii(data_dict):
    #         """
    #         Casts the values of dict to ascii
    #
    #         :param: input data dict
    #         :returns: ourput data dict
    #         """
    #         out_d = data_dict.copy()
    #         for key, value in data_dict.items():
    #             if isinstance(value, string_types):
    #                 out_d[key] = value.encode('ascii', 'ignore')
    #             elif isinstance(value, dict):
    #                 out_d[key] = JSONGrammar.cast_str_val_ascii(value)
    #         return out_d

    def add_description(self, description_dict):
        """
        Add a description to the properties

        :param description_dict: dictionary of descriptions,
             {name:meaning} structure
        """

        descr_filtered = {
            k: v for k, v in description_dict.items() if k in self._properties
        }
        self.schema.add_object({}, descr_filtered)

    def get_data_names(self):
        """Returns the list of data names

        :returns: the data names, as a dict keys set
        """
        return self._properties.keys()

    def is_data_name_existing(self, data_name):
        """Checks if data_name is present in grammar

        :param data_name: the data name
        :returns: True if data is in grammar
        """
        return data_name in self._properties

    def is_all_data_names_existing(self, data_names):
        """Checks if data_names are present in grammar

        :param data_names: the data names list
        :returns: True if all data are in grammar
        """
        exists = self.is_data_name_existing
        for data_name in data_names:
            if not exists(data_name):
                return False
        return True

    def update_from(self, input_grammar):
        """Adds properties coming from another grammar

        :param input_grammar: the grammar to take inputs from
        """
        if not isinstance(input_grammar, JSONGrammar):
            msg = "Cannot update grammar " "{} of type {} with {} of type {} ".format(
                self.name,
                type(self).__name__,
                input_grammar.name,
                type(input_grammar).__name__,
            )
            raise ValueError(msg)

        schema_dct = input_grammar.schema.to_dict()
        self.schema.add_schema(schema_dct)
        self._update_properties()

    def update_from_if_not_in(self, input_grammar, exclude_grammar):
        """Adds objects coming from input_grammar if they are not in
        exclude_grammar

        :param input_grammar: the grammar to take inputs from
        :param exclude_grammar: exclusion grammar
        """
        if isinstance(input_grammar, JSONGrammar) and isinstance(
            exclude_grammar, JSONGrammar
        ):
            in_schema_dct = input_grammar.schema.to_dict()
            in_schema_prop = in_schema_dct.get(self.PROPERTIES_FIELD, {})
            in_required = in_schema_dct.get(self.REQUIRED_FIELD, [])
            ex_schema_dct = exclude_grammar.schema.to_dict()
            ex_schema_prop = ex_schema_dct.get(self.PROPERTIES_FIELD, {})
            merged_required = []
            merged_prop = {}
            for prop_name, prop_schema in in_schema_prop.items():
                if prop_name not in ex_schema_prop:
                    merged_prop[prop_name] = prop_schema
                    if prop_name in in_required:
                        merged_required.append(prop_name)

            merged_schema = {
                self.TYPE_FIELD: self.OBJECT_FIELD,
                self.PROPERTIES_FIELD: merged_prop,
                self.REQUIRED_FIELD: merged_required,
            }

            self.schema.add_schema(merged_schema)
            self._update_properties()
        else:

            msg = "Cannot update grammar " + str(self.name)
            msg += " of type JSONGrammar with " + str(input_grammar.name)
            msg += " of type " + str(type(input_grammar).__name__)
            msg += " and " + str(exclude_grammar.name)
            msg += ", of type " + str(type(exclude_grammar).__name__)
            raise TypeError(msg)

    def restrict_to(self, data_names):
        """Restricts the grammar to a sublist of data names

        :param data_names: the names of the data to restrict the grammar to
        """

        self_schema_dct = self.schema.to_dict()
        self_schema_prop = self_schema_dct.get(self.PROPERTIES_FIELD, {})
        self_required = self_schema_dct.get(self.REQUIRED_FIELD, [])
        for prop_name in list(self_schema_prop.keys()):
            if prop_name not in data_names:
                del self_schema_prop[prop_name]
                if prop_name in self_required:
                    self_required.remove(prop_name)

        self_schema = {
            self.TYPE_FIELD: self.OBJECT_FIELD,
            self.PROPERTIES_FIELD: self_schema_prop,
            self.REQUIRED_FIELD: self_required,
        }

        self._init_schema()
        self.schema.add_schema(self_schema)
        self._update_properties()

    def remove_item(self, item_name):
        """Removes an item from the grammar

        :param item_name: the item name to be removed

        """
        if not self.is_data_name_existing(item_name):
            raise ValueError("Item " + str(item_name) + " not in grammar " + self.name)
        self_dict = self.schema.to_dict()
        del self_dict[self.PROPERTIES_FIELD][item_name]
        self_required = self_dict.get(self.REQUIRED_FIELD, [])
        if item_name in self_required:
            self_required.remove(item_name)
        self._init_schema()
        self.schema.add_schema(self_dict)
        self._update_properties()

    def set_item_value(self, item_name, item_value):
        """
        Sets the value of an item

        :param item_name: the item name to be modified
        :param item_value: value of the item
        """
        if not self.is_data_name_existing(item_name):
            raise ValueError("Item " + str(item_name) + " not in grammar " + self.name)
        self_dict = self.schema.to_dict()
        self_dict[self.PROPERTIES_FIELD][item_name] = item_value
        self._init_schema()
        self.schema.add_schema(self_dict)
        self._update_properties()

    def __getstate__(self):
        """
        Used by pickle to define what to serialize

        :returns: the dict to serialize
        """
        out_d = {}
        out_d.update(self.__dict__)
        out_d.pop("_validator")
        return out_d

    def __setstate__(self, data_dict):
        """
        Used by pickle to define what to deserialize

        :param data_dict : update self dict from data_dict to deserialize
        """
        self.__dict__.update(data_dict)
        self._update_properties()
