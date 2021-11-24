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
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
)

from numpy import generic, ndarray, zeros

from gemseo.core.grammars.abstract_grammar import AbstractGrammar
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.json_schema import MutableMappingSchemaBuilder
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.utils.py23_compat import PY2, JsonSchemaException, Path, compile_schema
from gemseo.utils.string_tools import MultiLineString

if PY2:
    import jsonschema
    from jsonschema import ValidationError
else:
    # TODO: remove when py27 is gone
    class ValidationError(BaseException):
        pass


LOGGER = logging.getLogger(__name__)

ElementType = Union[str, float, bool, Sequence[Union[str, float, bool]]]
NumPyNestedMappingType = Mapping[
    str, Union[ElementType, ndarray, generic, "NumPyNestedMappingType"]
]
MappingSchemaType = Dict[
    str, Union[ElementType, List[ElementType], "MappingSchemaType"]
]
DictSchemaType = Mapping[str, Union[ElementType, List[ElementType], "DictSchemaType"]]
SerializedGrammarType = Dict[
    str, Union[ElementType, List[ElementType], "SerializedGrammarType"]
]


class JSONGrammar(AbstractGrammar):
    """A grammar based on a JSON schema.

    Attributes:
        schema (MutableMappingSchemaBuilder): The JSON schema.
    """

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
        name,  # type: str
        schema_file=None,  # type: Optional[Union[str,Path]]
        schema=None,  # type: Optional[MappingSchemaType]
        descriptions=None,  # type: Optional[Mapping[str,str]]
    ):
        """
        Args:
            schema_file: The JSON schema file.
                If None, do not initialize the grammar from a JSON schema file.
            schema: A genson schema to initialize the grammar.
                If None,  do not initialize the grammar from a JSON schema.
            descriptions: The descriptions of the elements,
                in the form: ``{element_name: element_meaning}``.
                If None, use the descriptions available in the JSON schema if any.
        """
        super(JSONGrammar, self).__init__(name)
        self._validator = None
        self.schema = None
        self._schema_dict = None
        self._properties_dict = None
        self._init_schema()

        if schema is not None:
            self.schema.add_schema(schema)
        elif schema_file is not None:
            self.init_from_schema_file(schema_file, descriptions=descriptions)
        else:
            self.initialize_from_base_dict({}, description_dict=descriptions)

    def __repr__(self):  # type: (...) -> str
        return "{}, schema: {}".format(self, self.schema.to_json())

    def _init_schema(self):  # type: (...) -> None
        """Initialize the schema."""
        self.schema = MutableMappingSchemaBuilder()
        self._schema_dict = None
        self._properties_dict = None

    @property
    def schema_dict(self):  # type: (...) -> Dict[str,DictSchemaType]
        """The dictionary representation of the schema."""
        if self._schema_dict is None:
            self._schema_dict = self.schema.to_schema()
        return self._schema_dict

    @property
    def properties_dict(self):  # type: (...) -> Dict[str,DictSchemaType]
        """The dictionnary representation of the properties of the schema.

        Raises:
            ValueError: When the schema has no properties.
        """
        if self._properties_dict is None:
            self._properties_dict = self.schema_dict.get("properties")
            if self._properties_dict is None:
                raise ValueError(
                    "Schema has no properties: {}.".format(self.schema_dict)
                )
        return self._properties_dict

    def clear(self):  # type: (...) -> None
        self.__set_grammar_from_dict({})

    def _init_validator(self):  # type: (...) -> None
        """Initialize the validator."""
        self.schema_dict.pop("id", None)

        if PY2:
            # Use jsonschema instead of fastjsonschema when a property has anyOf.
            for value in self.schema_dict.get("properties", {}).values():
                if "anyOf" in value:
                    self._validator = jsonschema.validators.validator_for(
                        self.schema_dict
                    )(self.schema_dict).validate
                    return

        self._validator = compile_schema(self.schema_dict)

    @classmethod
    def cast_array_to_list(
        cls,
        data_dict,  # type: NumPyNestedMappingType
    ):  # type: (...) -> DictSchemaType
        """Cast the NumPy arrays to lists for dictionary values.

        Args:
            data_dict: The data mapping.

        Returns:
            The original mapping casted to a dictionary
            where NumPy arrays have been replaced with lists.
        """
        dict_of_list = dict(data_dict)
        for key, value in data_dict.items():
            if isinstance(value, (ndarray, generic)):
                dict_of_list[key] = value.real.tolist()
            elif isinstance(value, Mapping):
                dict_of_list[key] = cls.cast_array_to_list(value)

        return dict_of_list

    def is_required(
        self, element_name  # type: str
    ):  # type: (...) -> bool
        required_element_names = self.schema_dict.get("required", [element_name])
        return element_name in required_element_names

    def load_data(
        self,
        data,  # type: MutableMapping[str,ElementType]
        raise_exception=True,  # type: bool
    ):  # type: (...) -> MutableMapping[str,ElementType]
        """
        Raises:
            InvalidDataException:
                * If the passed data is not a dictionary.
                * If the data is not consistent with the grammar.
        """
        if not isinstance(data, MutableMapping):
            raise InvalidDataException(
                "Data must be a mutable mapping; "
                "got a {} instead.".format(type(data))
            )

        if self._validator is None:
            self._init_validator()

        data_to_check = self.cast_array_to_list(data)

        try:
            self._validator(data_to_check)
        except (JsonSchemaException, ValidationError) as error:
            log_message = MultiLineString()
            log_message.add("Invalid data in: {}".format(self.name))

            error_message = error.args[0]
            if error_message.startswith("data must contain"):
                # Error messages are not clear enough when missing elements
                # All keys are put in the message
                missing_elements = set(self.get_data_names()) - set(data.keys())

                if missing_elements:
                    log_message.add(
                        "Missing mandatory elements: {}".format(
                            ",".join(sorted(missing_elements))
                        )
                    )
                else:
                    log_message.add(", error: {}".format(error_message))
            else:
                log_message.add(", error: {}".format(error_message))

            LOGGER.error(log_message)

            if raise_exception:
                raise InvalidDataException(str(log_message))

            # Check a copy to keep types and arrays but store initial dict for complex
            # Add defaults
        for key, value in data_to_check.items():
            data.setdefault(key, value)

        return data

    def init_from_schema_file(
        self,
        schema_path,  # type: Union[str,Path]
        descriptions=None,  # type: Optional[Mapping[str,str]]
    ):  # type: (...) -> None
        """Set the grammar from a file.

        Args:
            schema_path: The path to the schema file.
            descriptions: The descriptions for the elements of the grammar,
                in the form: ``{element_name: element_meaning}``.
                If None, use the descriptions from the schema file.

        Raises:
            FileNotFoundError: If the schema file does not exist.
        """
        schema_path = Path(schema_path)

        if not schema_path.exists():
            raise FileNotFoundError(
                "Try to initialize grammar "
                "with not existing file: {}.".format(schema_path)
            )

        schema = json.loads(schema_path.read_text())
        self.__set_grammar_from_dict(schema, descriptions)

    def __set_grammar_from_dict(
        self,
        schema,  # type: Union[MappingSchemaType,MutableMappingSchemaBuilder]
        descriptions=None,  # type: Optional[Mapping[str,str]]
    ):  # type: (...) -> None
        """Set the grammar from a dictionary.

        Args:
            schema: The schema to set the grammar with.
            descriptions: The descriptions for the elements of the grammar,
                in the form: ``{element_name: element_meaning}``.
                If None, use the ``schema`` ones.
        """
        self._init_schema()
        self.__update_grammar_from_dict(schema, descriptions)

    def __update_grammar_from_dict(
        self,
        schema,  # type: Union[MappingSchemaType,MutableMappingSchemaBuilder]
        descriptions=None,  # type: Optional[Mapping[str,str]]
    ):  # type: (...) -> None
        """Update the grammar from a dictionary.

        Args:
            schema: The schema to update the grammar with.
            descriptions: The descriptions for the elements of the grammar,
                in the form: ``{element_name: element_meaning}``.
                If None, use the ``schema`` ones.
        """
        if descriptions is not None:
            if not isinstance(schema, dict):
                schema = schema.to_schema()

            for property_name, property_schema in schema["properties"].items():
                descr = descriptions.get(property_name)
                if descr is not None:
                    self.__add_description_to_types(descr, property_schema)

        self.__merge_schema(schema)

    def __add_description_to_types(
        self,
        description,  # type: str
        property_schema,  # type: Mapping[str, str]
    ):  # type: (...) -> None
        """Add the description for all the types found in the schema of a parameter.

        Args:
            description: The description of the parameter.
            property_schema: The schema of the parameter.
        """

        if "anyOf" in property_schema:
            for each_type in property_schema["anyOf"]:
                each_type["description"] = description
        else:
            property_schema["description"] = description

    def __merge_schema(
        self,
        schema,  # type: MappingSchemaType
    ):  # type: (...) -> None
        """Merge a schema in the current one.

        Args:
            schema: The schema to be merge, could be a schema object or a dictionary.
        """
        self.schema.add_schema(schema)
        self.__reset_schema_attrs()

    def initialize_from_data_names(
        self,
        data_names,  # type: Iterable[str]
        descriptions=None,  # type: Optional[Mapping[str,str]]
    ):  # type: (...) -> None
        """Initialize the grammar from the names and descriptions of the elements.

        Use float type.

        Args:
            descriptions: The descriptions of the elements,
                in the form: ``{element_name: element_meaning}``.
                If None, do not initialize the elements with descriptions.
        """
        element_value = zeros(1)
        elements_values = {element_name: element_value for element_name in data_names}
        self.initialize_from_base_dict(elements_values, description_dict=descriptions)

    def initialize_from_base_dict(
        self,
        typical_data_dict,  # type: Mapping[str,ElementType]
        description_dict=None,  # type: Optional[Mapping[str,str]]
    ):  # type: (...) -> None
        """Initialize the grammar with types and names from a typical data entry.

        The keys of the ``typical_data_dict`` are the names of the elements.
        The types of the values of the ``typical_data_dict`` will be converted
        to JSON Schema types and define the elements of the JSON Schema.

        Args:
            description_dict: The descriptions of the data names,
                in the form: ``{element_name: element_meaning}``.
                If None, do not initialize the elements with descriptions.
        """
        # Convert arrays to list as for check
        list_data_dict = self.cast_array_to_list(typical_data_dict)
        self.schema.add_object(list_data_dict)
        self.__set_grammar_from_dict(self.schema, description_dict)

    def get_data_names(self):  # type: (...) -> List[str]
        return list(self.schema.keys())

    def is_data_name_existing(
        self,
        data_name,  # type: str
    ):  # type: (...) -> bool
        return data_name in self.schema._properties

    def is_type_array(
        self, data_name  # type: str
    ):  # type: (...) -> bool
        if not self.is_data_name_existing(data_name):
            raise ValueError("{} is not in the grammar.".format(data_name))
        prop = self.properties_dict.get(data_name)
        return "array" == prop.get("type")

    def is_all_data_names_existing(
        self,
        data_names,  # type: Iterable[str]
    ):  # type: (...) -> bool
        properties = self.schema._properties
        for data_name in data_names:
            if data_name not in properties:
                return False
        return True

    def update_from(
        self,
        input_grammar,  # type: JSONGrammar
    ):  # type: (...) -> None
        """
        Raises:
            TypeError: If the passed grammar is not a JSONGrammar.
        """
        if not isinstance(input_grammar, JSONGrammar):
            msg = (
                "A {} is expected as input, but an object of type {} "
                "has been provided.".format(self.__class__, type(input_grammar))
            )
            raise TypeError(msg)

        self.__merge_schema(input_grammar.schema)

    def to_simple_grammar(self):  # type: (...) -> SimpleGrammar
        """Convert to the base :class:`.SimpleGrammar` type.

        Ignore the features of JSONGrammar that are not supported by SimpleGrammar.

        Returns:
            A :class:`.SimpleGrammar` equivalent to the current grammar.
        """
        grammar = SimpleGrammar(self.name)
        schema_dict = self.schema_dict
        properties = schema_dict.get(self.PROPERTIES_FIELD, {})

        names_to_types = {}
        for property_name, property_description in properties.items():
            property_json_type = property_description.get("type")
            if property_json_type not in self.TYPES_MAP:
                property_type = None
            else:
                property_type = self.TYPES_MAP[property_description["type"]]

            names_to_types[property_name] = property_type

            if property_json_type == "array" and "items" in property_description:
                property_json_sub_type = property_description["items"].get("type")
                if property_json_sub_type not in ["number", "integer", None]:
                    message = (
                        "Unsupported type '{}' in JSONGrammar '{}' "
                        "for property '{}' in conversion to simple grammar."
                    ).format(property_json_sub_type, self.name, property_name)
                    LOGGER.warning(message)

            for feature in ["minItems", "maxItems", "additionalItems", "contains"]:
                if feature in property_description:
                    message = (
                        "Unsupported feature '{}' in JSONGrammar '{}' "
                        "for property '{}' in conversion to simple grammar."
                    ).format(feature, self.name, property_name)
                    LOGGER.warning(message)

        grammar.update_elements(**names_to_types)

        required_data_names = self.schema_dict.get("required", [])
        grammar.update_required_elements(**dict.fromkeys(required_data_names, True))

        optional_data_names = set(self.schema_dict.get("properties", [])) - set(
            required_data_names
        )
        grammar.update_required_elements(**dict.fromkeys(optional_data_names, False))

        return grammar

    def update_from_if_not_in(
        self,
        input_grammar,  # type: JSONGrammar
        exclude_grammar,  # type: JSONGrammar
    ):  # type: (...) -> None
        if not (
            isinstance(input_grammar, self.__class__)
            and isinstance(exclude_grammar, self.__class__)
        ):
            msg = self._get_update_error_msg(self, input_grammar, exclude_grammar)
            raise TypeError(msg)

        schema = MutableMappingSchemaBuilder()
        schema.add_schema(input_grammar.schema)

        for element_name in exclude_grammar.schema.keys():
            try:
                del schema[element_name]
            except KeyError:
                pass

        self.__merge_schema(schema)

    def restrict_to(
        self,
        data_names,  # type: Sequence[str]
    ):  # type: (...) -> None
        for element_name in list(self.schema.keys()):
            if element_name not in data_names:
                del self.schema[element_name]
        self.__reset_schema_attrs()

    def remove_item(
        self,
        item_name,  # type: str
    ):  # type: (...) -> None
        del self.schema[item_name]
        self.__reset_schema_attrs()

    def __reset_schema_attrs(self):  # type: (...) -> None
        """Resets the validator, properties dict and schema dict conversions."""
        self._validator = None
        self._properties_dict = None
        self._schema_dict = None

    def set_item_value(
        self,
        item_name,  # type: str
        item_value,  # type: Dict[str,str]
    ):  # type: (...) -> None
        """Set the value of an element.

        Args:
            item_name: The name of the element.
            item_value: The value of the element.

        Raises:
            ValueError: If the item is not in the grammar.
        """
        if not self.is_data_name_existing(item_name):
            raise ValueError("Item {} not in grammar {}.".format(item_name, self.name))
        schema = self.schema_dict
        schema[self.PROPERTIES_FIELD][item_name] = item_value

        self.__set_grammar_from_dict(schema)

    def write_schema(
        self,
        path=None,  # type: Optional[Path,str]
    ):  # type: (...) -> None
        """Write the schema to a file.

        Args:
            path: The file path.
                If None,
                then write to a file named after the grammar and with .json extension.
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

    def __getstate__(self):  # type: (...) -> SerializedGrammarType
        """Used by pickle to define what to serialize.

        Returns:
            The dict to serialize.
        """
        deserialized_grammar = dict(self.__dict__)
        deserialized_grammar.pop("_validator")
        # genson schema cannot be pickled: use its dictionary representation
        deserialized_grammar["schema"] = self.schema_dict
        return deserialized_grammar

    def __setstate__(
        self,
        serialized_grammar,  # type: SerializedGrammarType
    ):  # type: (...) -> None
        """Used by pickle to define what to deserialize.

        Args:
            data_dict: update self dict from data_dict to deserialize.
        """
        self.__dict__.update(serialized_grammar)
        # genson schema cannot be pickled: use its dictionary representation
        self.__set_grammar_from_dict(serialized_grammar.pop("schema"))
