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
from __future__ import annotations

import json
import logging
from numbers import Number
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import Sequence
from typing import Union

from fastjsonschema import compile as compile_schema
from fastjsonschema.exceptions import JsonSchemaException
from numpy import generic
from numpy import ndarray
from numpy import zeros

from gemseo.core.grammars.abstract_grammar import AbstractGrammar
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.json_schema import MutableMappingSchemaBuilder
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.utils.string_tools import MultiLineString

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
        name: str,
        schema_file: str | Path | None = None,
        schema: MappingSchemaType | None = None,
        descriptions: Mapping[str, str] | None = None,
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
        super().__init__(name)
        self._validator = None
        self.schema = None
        self._schema_dict = None
        self._properties_dict = None
        self.__data_names = None
        self.__data_names_keyset = None
        self._init_schema()

        if schema is not None:
            self.schema.add_schema(schema)
        elif schema_file is not None:
            self.init_from_schema_file(schema_file, descriptions=descriptions)
        else:
            self.initialize_from_base_dict({}, description_dict=descriptions)

    def __repr__(self) -> str:
        return f"{self}, schema: {self.schema.to_json()}"

    def _init_schema(self) -> None:
        """Initialize the schema."""
        self.schema = MutableMappingSchemaBuilder()
        self._schema_dict = None
        self._properties_dict = None
        self.__data_names = None
        self.__data_names_keyset = None

    @property
    def schema_dict(self) -> dict[str, DictSchemaType]:
        """The dictionary representation of the schema."""
        if self._schema_dict is None:
            self._schema_dict = self.schema.to_schema()
        return self._schema_dict

    @property
    def data_names(self) -> list[str]:
        """The data names of the grammar."""
        if self.__data_names is None:
            self.__data_names = list(self.data_names_keyset)
        return self.__data_names

    @property
    def data_names_keyset(self) -> Iterable[str]:
        """The data names of the grammar as dict_keys."""
        if self.__data_names_keyset is None:
            try:
                self.__data_names_keyset = self.properties_dict
            except ValueError:
                self.__data_names_keyset = {}
        return self.__data_names_keyset

    @property
    def properties_dict(self) -> dict[str, DictSchemaType]:
        """The dictionnary representation of the properties of the schema.

        Raises:
            ValueError: When the schema has no properties.
        """
        if self._properties_dict is None:
            self._properties_dict = self.schema_dict.get("properties")
            if self._properties_dict is None:
                raise ValueError(f"Schema has no properties: {self.schema_dict}.")
        return self._properties_dict

    def clear(self) -> None:
        self.__set_grammar_from_dict({})

    def _init_validator(self) -> None:
        """Initialize the validator."""
        self.schema_dict.pop("id", None)
        self._validator = compile_schema(self.schema_dict)

    @classmethod
    def cast_array_to_list(
        cls,
        data_dict: NumPyNestedMappingType,
    ) -> DictSchemaType:
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

    def is_required(self, element_name: str) -> bool:
        required_element_names = self.schema_dict.get("required", [element_name])
        return element_name in required_element_names

    def load_data(
        self,
        data: MutableMapping[str, ElementType],
        raise_exception: bool = True,
    ) -> MutableMapping[str, ElementType]:
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
        except JsonSchemaException as error:
            log_message = MultiLineString()
            log_message.add(f"Invalid data in: {self.name}")

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
                    log_message.add(f", error: {error_message}")
            else:
                log_message.add(f", error: {error_message}")

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
        schema_path: str | Path,
        descriptions: Mapping[str, str] | None = None,
    ) -> None:
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
        schema: MappingSchemaType | MutableMappingSchemaBuilder,
        descriptions: Mapping[str, str] | None = None,
    ) -> None:
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
        schema: MappingSchemaType | MutableMappingSchemaBuilder,
        descriptions: Mapping[str, str] | None = None,
    ) -> None:
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
        description: str,
        property_schema: Mapping[str, str],
    ) -> None:
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
        schema: MappingSchemaType,
    ) -> None:
        """Merge a schema in the current one.

        Args:
            schema: The schema to be merge, could be a schema object or a dictionary.
        """
        self.schema.add_schema(schema)
        self.__reset_schema_attrs()

    def initialize_from_data_names(
        self,
        data_names: Iterable[str],
        descriptions: Mapping[str, str] | None = None,
    ) -> None:
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
        typical_data_dict: Mapping[str, ElementType],
        description_dict: Mapping[str, str] | None = None,
    ) -> None:
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

    def get_data_names(self) -> list[str]:
        return self.data_names

    def is_data_name_existing(
        self,
        data_name: str,
    ) -> bool:
        return data_name in self.schema._properties

    def is_type_array(self, data_name: str) -> bool:
        if not self.is_data_name_existing(data_name):
            raise ValueError(f"{data_name} is not in the grammar.")
        prop = self.properties_dict.get(data_name)
        return "array" == prop.get("type")

    def is_all_data_names_existing(
        self,
        data_names: Iterable[str],
    ) -> bool:
        properties = self.schema._properties
        for data_name in data_names:
            if data_name not in properties:
                return False
        return True

    def update_from(
        self,
        input_grammar: JSONGrammar,
    ) -> None:
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

    def to_simple_grammar(self) -> SimpleGrammar:
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
        input_grammar: JSONGrammar,
        exclude_grammar: JSONGrammar,
    ) -> None:
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
        data_names: Sequence[str],
    ) -> None:
        for element_name in list(self.schema.keys()):
            if element_name not in data_names:
                del self.schema[element_name]
        self.__reset_schema_attrs()

    def remove_item(
        self,
        item_name: str,
    ) -> None:
        del self.schema[item_name]
        self.__reset_schema_attrs()

    def __reset_schema_attrs(self) -> None:
        """Resets the validator, properties dict and schema dict conversions."""
        self._validator = None
        self._properties_dict = None
        self._schema_dict = None
        self.__data_names = None
        self.__data_names_keyset = None

    def set_item_value(
        self,
        item_name: str,
        item_value: dict[str, str],
    ) -> None:
        """Set the value of an element.

        Args:
            item_name: The name of the element.
            item_value: The value of the element.

        Raises:
            ValueError: If the item is not in the grammar.
        """
        if not self.is_data_name_existing(item_name):
            raise ValueError(f"Item {item_name} not in grammar {self.name}.")
        schema = self.schema_dict
        schema[self.PROPERTIES_FIELD][item_name] = item_value

        self.__set_grammar_from_dict(schema)

    def write_schema(
        self,
        path: Path | str | None = None,
    ) -> None:
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

        json.dump(
            self.schema.to_json(),
            path.open("w", encoding="utf-8"),
        )

    def __getstate__(self) -> SerializedGrammarType:
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
        serialized_grammar: SerializedGrammarType,
    ) -> None:
        """Used by pickle to define what to deserialize.

        Args:
            data_dict: update self dict from data_dict to deserialize.
        """
        self.__dict__.update(serialized_grammar)
        # genson schema cannot be pickled: use its dictionary representation
        self.__set_grammar_from_dict(serialized_grammar.pop("schema"))
