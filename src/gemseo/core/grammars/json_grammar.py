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
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Sequence
from typing import Union

from fastjsonschema import compile as compile_schema
from fastjsonschema import JsonSchemaException
from numpy import generic
from numpy import ndarray

from gemseo.core.discipline_data import Data
from gemseo.core.discipline_data import MutableData
from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.json_schema import MutableMappingSchemaBuilder
from gemseo.core.grammars.simple_grammar import NamesToTypes
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)

ElementType = Union[str, float, bool, Sequence[Union[str, float, bool]]]
DictSchemaType = Mapping[str, Union[ElementType, List[ElementType], "DictSchemaType"]]
SerializedGrammarType = Dict[
    str, Union[ElementType, List[ElementType], "SerializedGrammarType"]
]


class JSONGrammar(BaseGrammar):
    """A grammar based on a JSON schema."""

    __validator: Callable[[Mapping[str, Any]], None] | None
    """The schema validator."""

    __schema: dict[str, Any]
    """The schema stored as a dictionary."""

    __schema_builder: MutableMappingSchemaBuilder
    """The internal schema object."""

    __JSON_TO_PYTHON_TYPES = {
        "array": ndarray,
        "string": str,
        "integer": int,
        "boolean": bool,
        "number": Number,
        "float": Number,
    }
    """The binding from JSON types to Python types."""

    __NUMERIC_TYPE_NAMES: ClassVar[tuple[str]] = ("number", "float", "integer")

    def __init__(
        self,
        name: str,
        schema_path: str | Path | None = None,
        descriptions: Mapping[str, str] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            schema_path: The path to a JSON schema file.
                If ``None``, do not initialize the grammar from a JSON schema file.
            descriptions: The descriptions of the elements read from ``schema_path``,
                in the form: ``{element_name: element_meaning}``.
                If ``None``, use the descriptions available in the JSON schema if any.
            **kwargs: These arguments are not used.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        if schema_path is not None:
            self.update_from_file(schema_path)
            self.set_descriptions(descriptions)

    def __delitem__(
        self,
        name: str,
    ) -> None:
        del self.__schema_builder[name]
        self.__init_dependencies()

    def __getitem__(self, name: str) -> Any:
        return self.__schema_builder[name]

    def __len__(self) -> int:
        return len(self.__schema_builder)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.__schema_builder)

    def rename_element(self, current_name: str, new_name: str) -> None:  # noqa: D102
        self.__schema_builder.properties[
            new_name
        ] = self.__schema_builder.properties.pop(current_name)

        required = self.__schema_builder.required
        if current_name in required:
            required.remove(current_name)
            required.add(new_name)
        self.__init_dependencies()

    def update(
        self,
        grammar: JSONGrammar | Iterable[str] | DictSchemaType,
        exclude_names: Iterable[str] | None = None,
    ) -> None:
        """Update the elements from another grammar or names or a schema."""
        if isinstance(grammar, JSONGrammar):
            if exclude_names:
                schema_builder = deepcopy(grammar.__schema_builder)
                for name in exclude_names:
                    if name in schema_builder:
                        del schema_builder[name]
            else:
                schema_builder = grammar.__schema_builder
            self.__update(schema_builder)
            self._update_namespaces_from_grammar(grammar)
        elif isinstance(grammar, BaseGrammar):
            raise TypeError(
                f"A JSONGrammar cannot be updated from a grammar of type: {type(grammar)}"
            )
        elif isinstance(grammar, Mapping):
            if exclude_names:
                schema = dict(grammar)
                for name in exclude_names:
                    schema.pop(name)
            else:
                schema = grammar
            self.__update(schema)
        elif isinstance(grammar, Iterable):
            if not grammar:
                return
            if exclude_names is None:
                exclude_names = tuple()
            for name in grammar:
                if name not in exclude_names:
                    self.__schema_builder.add_object({name: [0.0]})
        else:
            raise TypeError()
        self.__init_dependencies()

    def clear(self) -> None:  # noqa: D102
        self.__schema_builder = MutableMappingSchemaBuilder()
        self.__init_dependencies()

    def __repr__(self) -> str:
        return f"{self}, schema: {self.__schema_builder.to_json(indent=2)}"

    def validate(
        self,
        data: MutableData,
        raise_exception: bool = True,
    ) -> None:
        """
        Raises:
            InvalidDataException: If the passed data is not a dictionary,
                or if the data is not consistent with the grammar.
        """  # noqa: D205, D212, D415
        error_message = MultiLineString()

        # Check the required names explicitly to provide a clearer message.
        missing_names = self.required_names - set(data.keys())
        if missing_names:
            error_message.add(
                "Missing required names: {}.".format(",".join(sorted(missing_names)))
            )

        if self.__validator is None:
            self._create_validator()

        data_to_check = self.__cast_array_to_list(data)

        try:
            self.__validator(data_to_check)
        except JsonSchemaException as error:
            if not error.args[0].startswith("data must contain"):
                error_message.add(f", error: {error.args[0]}")
            LOGGER.error(error_message)
            if raise_exception:
                raise InvalidDataException(str(error_message))

    def update_from_data(
        self,
        data: Data,
    ) -> None:
        """
        Notes:
            The types of the values of the ``data`` will be converted
            to JSON Schema types and define the elements of the JSON Schema.
        """  # noqa: D205, D212, D415
        if not data:
            return
        self.__schema_builder.add_object(self.__cast_array_to_list(data))
        self.__init_dependencies()

    def is_array(  # noqa: D102
        self,
        name: str,
        numeric_only: bool = False,
    ) -> bool:
        self._check_name(name)

        prop = self.schema.get("properties").get(name)
        if prop.get("type") != "array":
            return False
        if numeric_only:
            return self.__is_array_of_numeric_value(prop)
        return True

    @staticmethod
    def __is_array_of_numeric_value(prop: Any) -> bool:
        """Whether the array (which can be nested) contains numeric values at the end.

        This method is recursive in order to be able to take into account nested arrays.

        Args:
            prop: The grammar property.

        Returns:
            Whether the property contains numeric values at the end.
        """
        sub_prop = prop.get("items")
        # If the sub_prob is not defined, we assume that it is a numeric value
        if sub_prop is None:
            return True
        sub_prop_type = sub_prop.get("type")
        if sub_prop_type == "array":
            return JSONGrammar.__is_array_of_numeric_value(sub_prop)
        return sub_prop.get("type") in JSONGrammar.__NUMERIC_TYPE_NAMES

    def restrict_to(  # noqa: D102
        self,
        names: Sequence[str],
    ) -> None:
        self._check_name(*names)
        for element_name in tuple(self.__schema_builder):
            if element_name not in names:
                del self.__schema_builder[element_name]
        self.__init_dependencies()

    def convert_to_simple_grammar(self) -> SimpleGrammar:  # noqa: D102
        grammar = SimpleGrammar(self.name)
        grammar.update(self.__get_names_to_types())
        for name in set(self.keys()) - self.required_names:
            grammar.required_names.remove(name)
        return grammar

    @property
    def required_names(self) -> set[str]:  # noqa: D102
        return self.__schema_builder.required

    # API not in the base class.

    def __update(
        self,
        schema: DictSchemaType,
    ) -> None:
        """Update the elements from a schema.

        Args:
            schema: The schema to take the elements from.
        """
        self.__schema_builder.add_schema(schema)

    def update_from_file(
        self,
        path: str | Path,
    ) -> None:
        """Update the grammar from a schema file.

        Args:
            path: The path to the schema file.

        Raises:
            FileNotFoundError: If the schema file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot update the grammar from non existing file: {path}."
            )
        self.__schema_builder.add_schema(json.loads(path.read_text()))
        self.__init_dependencies()

    def write(
        self,
        path: Path | str | None = None,
    ) -> None:
        """Write the grammar schema to a file.

        Args:
            path: The file path.
                If ``None``,
                then write to a file named after the grammar and with .json extension.
        """
        if path is None:
            path = Path(self.name).with_suffix(".json")
        else:
            path = Path(path)
        path.write_text(self.__schema_builder.to_json(indent=2), encoding="utf-8")

    def to_json(self, *args, **kwargs) -> str:
        """Return the JSON representation of the grammar schema.

        Args:
            *args: The positional arguments passed to :func:`json.dumps`.
            **kwargs: The keyword arguments passed to :func:`json.dumps`.

        Returns:
            The JSON representation of the schema.
        """
        return self.__schema_builder.to_json(*args, **kwargs)

    @property
    def schema(self) -> dict[str, DictSchemaType]:
        """The dictionary representation of the schema."""
        if not self.__schema:
            self.__schema = self.__schema_builder.to_schema()
        return self.__schema

    def _create_validator(self) -> None:
        """Create the schema validator."""
        self.schema.pop("id", None)
        self.__validator = compile_schema(self.schema)

    def set_descriptions(self, descriptions: Mapping[str, str]) -> None:
        """Set the properties descriptions.

        Args:
            descriptions: The descriptions, mapping properties names
                to the description.
        """
        if not descriptions:
            return

        for property_name, property_schema in self.__schema_builder.properties.items():
            description = descriptions.get(property_name)
            if description:
                schema = property_schema.to_schema()
                schema["description"] = description
                property_schema.add_schema(schema)

        self.__init_dependencies()

    @classmethod
    def __cast_array_to_list(
        cls,
        data: Data,
    ) -> DictSchemaType:
        """Cast the NumPy arrays to lists for dictionary values.

        Args:
            data: The data mapping.

        Returns:
            The original mapping cast to a dictionary
            where NumPy arrays have been replaced with lists.
        """
        dict_of_list = dict(data)
        for key, value in data.items():
            if isinstance(value, (ndarray, generic)):
                dict_of_list[key] = value.real.tolist()
            elif isinstance(value, Mapping):
                dict_of_list[key] = cls.__cast_array_to_list(value)
        return dict_of_list

    def __get_names_to_types(self) -> NamesToTypes:
        """Create the mapping from element names to elements types.

        Returns:
            The mapping from element names to elements types.
        """
        properties = self.schema.get("properties")

        if properties is None:
            return {}

        names_to_types = {}

        for property_name, property_description in properties.items():
            property_json_type = property_description.get("type")

            self.__warn_for_array(
                property_name, property_json_type, property_description
            )
            self.__warn_for_items(property_name, property_description)

            if property_json_type not in self.__JSON_TO_PYTHON_TYPES:
                property_type = None
            else:
                property_type = self.__JSON_TO_PYTHON_TYPES[
                    property_description["type"]
                ]

            names_to_types[property_name] = property_type

        return names_to_types

    def __warn_for_array(
        self,
        property_name: str,
        property_json_type: str,
        property_description: Mapping[str, Mapping[str, str | None]],
    ) -> None:
        """Log a warning when an array has unsupported types.

        Args:
            property_name: The name of the property.
            property_json_type: The json type of the property.
            property_description: The description of the property.
        """
        if property_json_type == "array" and "items" in property_description:
            property_json_sub_type = property_description["items"].get("type")
            if property_json_sub_type not in ["number", "integer", None]:
                message = (
                    "Unsupported type '%s' in JSONGrammar '%s' "
                    "for property '%s' in conversion to simple grammar."
                )
                LOGGER.warning(
                    message, property_json_sub_type, self.name, property_name
                )

    def __warn_for_items(
        self,
        property_name: str,
        property_description: Iterable[str],
    ) -> None:
        """Log a warning when an item has unsupported descriptions.

        Args:
            property_name: The name of the property.
            property_description: The description of the property.
        """
        for feature in ["minItems", "maxItems", "additionalItems", "contains"]:
            if feature in property_description:
                message = (
                    "Unsupported feature '%s' in JSONGrammar '%s' "
                    "for property '%s' in conversion to simple grammar."
                )
                LOGGER.warning(message, feature, self.name, property_name)

    def __init_dependencies(self) -> None:
        """Resets the validator and schema dict."""
        self.__validator = None
        self.__schema = {}

    def _check_name(self, *names: str) -> None:
        for name in names:
            if name not in self.__schema_builder:
                raise KeyError(f"The name {name} is not in the grammar.")

    def __getstate__(self) -> SerializedGrammarType:
        # Ensure self.__schema is filled.
        self.schema
        state = dict(self.__dict__)
        # The validator will be recreated on demand.
        del state[f"_{self.__class__.__name__}__validator"]
        # The schema builder cannot be pickled.
        del state[f"_{self.__class__.__name__}__schema_builder"]
        return state

    def __setstate__(
        self,
        state: SerializedGrammarType,
    ) -> None:
        self.__dict__.update(state)
        # That will create the missing attributes.
        self.clear()
        self.__schema_builder.add_schema(state[f"_{self.__class__.__name__}__schema"])
