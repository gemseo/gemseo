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
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from copy import copy
from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import Union

from fastjsonschema import JsonSchemaException
from fastjsonschema import compile as compile_schema
from numpy import generic
from numpy import ndarray

from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.core.grammars.base_grammar import NamesToTypes
from gemseo.core.grammars.json_schema import MutableMappingSchemaBuilder
from gemseo.core.grammars.simple_grammar import SimpleGrammar

if TYPE_CHECKING:
    from gemseo.core.discipline_data import Data
    from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)

ElementType = Union[str, float, bool, Sequence[Union[str, float, bool]]]
DictSchemaType = Mapping[str, Union[ElementType, list[ElementType], "DictSchemaType"]]
SerializedGrammarType = dict[
    str, Union[ElementType, list[ElementType], "SerializedGrammarType"]
]


class JSONGrammar(BaseGrammar):
    """A grammar based on a JSON schema.

    For the dictionary-like methods similar to ``update``,
    when a key exists in both grammars,
    the values can be merged instead of being
    updated by passing ``merge = True``.
    In that case, the resulting grammar will allow any of the values.
    """

    DATA_CONVERTER_CLASS: ClassVar[str] = "JSONGrammarDataConverter"

    __validator: Callable[[Mapping[str, Any]], None] | None
    """The schema validator."""

    __schema: dict[str, Any]
    """The schema stored as a dictionary."""

    __schema_builder: MutableMappingSchemaBuilder
    """The internal schema object."""

    __JSON_TO_PYTHON_TYPES: Final[dict[str, type]] = {
        "array": ndarray,
        "string": str,
        "integer": int,
        "boolean": bool,
        "number": complex,
    }
    """The mapping from JSON types to Python types."""

    __PYTHON_TO_JSON_TYPES: Final[dict[type, str]] = {
        ndarray: "array",
        list: "array",
        tuple: "array",
        str: "string",
        int: "integer",
        bool: "boolean",
        complex: "number",
        float: "number",
    }
    """The mapping from Python types to JSON types."""

    __WARNING_TEMPLATE: Final[str] = (
        "Unsupported %s '%s' in JSONGrammar '%s' "
        "for property '%s' in conversion to SimpleGrammar."
    )
    """The logging warning template for conversion to SimpleGrammar."""

    def __init__(
        self,
        name: str,
        file_path: str | Path = "",
        descriptions: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            file_path: The path to a JSON schema file.
                If ``None``, do not initialize the grammar from a JSON schema file.
            descriptions: The descriptions of the elements read from ``file_path``,
                in the form: ``{element_name: element_meaning}``.
                If ``None``, use the descriptions available in the JSON schema if any.
            **kwargs: These arguments are not used.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        if file_path:
            self.update_from_file(file_path)
            self.set_descriptions(descriptions)

    def __getitem__(self, name: str) -> Any:
        return self.__schema_builder[name]

    def __len__(self) -> int:
        return len(self.__schema_builder)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.__schema_builder)

    def _delitem(self, name: str) -> None:  # noqa:D102
        del self.__schema_builder[name]
        self.__init_dependencies()

    def _copy(self, grammar: JSONGrammar) -> None:
        # Updating is much faster than deep copying a schema builder.
        grammar.__schema_builder.add_schema(self.__schema_builder, True)
        grammar.__schema = self.__schema.copy()
        grammar.__validator = copy(self.__validator)

    def _rename_element(self, current_name: str, new_name: str) -> None:  # noqa: D102
        self.__schema_builder.properties[new_name] = (
            self.__schema_builder.properties.pop(current_name)
        )
        required = self.__schema_builder.required
        if current_name in required:
            required.remove(current_name)
            required.add(new_name)
        self.__init_dependencies()

    def update(
        self,
        grammar: JSONGrammar,
        exclude_names: Iterable[str] = (),
        merge: bool = False,
    ) -> None:
        """Update the elements from another grammar or names or a schema.

        Args:
            merge: Whether to merge or update the grammar.

        Raises:
            TypeError: If the grammar is not a :class:`JSONGrammar`.
        """  # noqa:D417
        if not isinstance(grammar, JSONGrammar):
            msg = (
                "A JSONGrammar cannot be updated from a grammar of type: "
                f"{type(grammar)}"
            )
            raise TypeError(msg)

        if not grammar:
            return

        if exclude_names:
            schema_builder = deepcopy(grammar.__schema_builder)
            for name in exclude_names:
                if name in schema_builder:
                    del schema_builder[name]
        else:
            schema_builder = grammar.__schema_builder

        self.__schema_builder.add_schema(schema_builder, not merge)
        self.__init_dependencies()
        super().update(grammar, exclude_names)

    def update_from_names(
        self,
        names: Iterable[str],
        merge: bool = False,
    ) -> None:
        """
        Args:
            merge: Whether to merge or update the grammar.
        """  # noqa: D205, D212, D415
        if not names:
            return
        for name in names:
            self.__schema_builder.add_object({name: [0.0]}, not merge)
        self.__init_dependencies()

    def update_from_data(
        self,
        data: Data,
        merge: bool = False,
    ) -> None:
        """
        Args:
            merge: Whether to merge or update the grammar.

        Notes:
            The types of the values of the ``data`` will be converted
            to JSON Schema types and define the elements of the JSON Schema.
        """  # noqa: D205, D212, D415
        if not data:
            return
        self.__schema_builder.add_object(self.__cast_data_mapping(data), not merge)
        self.__init_dependencies()

    def update_from_types(  # noqa: D102
        self,
        names_to_types: Mapping[str, type],
        merge: bool = False,
    ) -> None:
        if not names_to_types:
            return

        try:
            properties = {
                element_name: {"type": self.__PYTHON_TO_JSON_TYPES[element_type]}
                for element_name, element_type in names_to_types.items()
            }
        except KeyError as error:
            raise KeyError(
                f"Unsupported python type for a JSON Grammar: {error}"
            ) from None

        schema = {
            "$schema": "http://json-schema.org/draft-04/schema",
            "type": "object",
            "properties": properties,
            "required": list(names_to_types.keys()),
        }
        self.__schema_builder.add_schema(schema, not merge)
        self.__init_dependencies()

    def _clear(self) -> None:  # noqa: D102
        self.__schema_builder = MutableMappingSchemaBuilder()
        self.__init_dependencies()

    def _update_grammar_repr(self, repr_: MultiLineString, properties: Any) -> None:
        for k, v in properties.to_schema().items():
            self.__repr_property(k, v, repr_)

    @classmethod
    def __repr_property(cls, name: str, value: Any, repr_: MultiLineString) -> None:
        """Update the string representation of the grammar with that of a property.

        Args:
            name: The name of the property.
            value: The value of the property.
            repr_: The string representation of the grammar.
        """
        if isinstance(value, Mapping):
            repr_.add(f"{name.capitalize()}:")
            repr_.indent()
            for k, v in value.items():
                cls.__repr_property(k, v, repr_)
            repr_.dedent()
        else:
            repr_.add(f"{name.capitalize()}: {value}")

    def _validate(  # noqa:D102
        self,
        data: Data,
        error_message: MultiLineString,
    ) -> bool:
        if self.__validator is None:
            self._create_validator()

        try:
            self.__validator(self.__cast_data_mapping(data))
        except JsonSchemaException as error:
            if not error.args[0].startswith("data must contain"):
                error_message.add(f"error: {error.args[0]}")
            return False

        return True

    def is_array(  # noqa: D102
        self,
        name: str,
        numeric_only: bool = False,
    ) -> bool:
        self._check_name(name)
        if numeric_only:
            return self.data_converter.is_numeric(name)
        return self.schema.get("properties").get(name).get("type") == "array"

    def _restrict_to(  # noqa: D102
        self,
        names: Iterable[str],
    ) -> None:
        for element_name in self.__schema_builder.keys() - names:
            del self.__schema_builder[element_name]
        self.__init_dependencies()

    def to_simple_grammar(self) -> SimpleGrammar:  # noqa: D102
        grammar = SimpleGrammar(self.name)
        grammar.update_from_types(self.__get_names_to_types())
        for name in self.keys() - self.required_names:
            grammar.required_names.remove(name)
        grammar._defaults.update(self._defaults)
        return grammar

    @property
    def required_names(self) -> set[str]:  # noqa: D102
        return self.__schema_builder.required

    # API not in the base class.

    def update_from_file(self, path: str | Path, merge: bool = False) -> None:
        """Update the grammar from a schema file.

        Args:
            path: The path to the schema file.
            merge: Whether to merge or update the grammar.

        Raises:
            FileNotFoundError: If the schema file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot update the grammar from non existing file: {path}."
            )
        self.__schema_builder.add_schema(json.loads(path.read_text()), not merge)
        self.__init_dependencies()

    def update_from_schema(self, schema: DictSchemaType, merge: bool = False) -> None:
        """Update the grammar from a json schema.

        Args:
            schema: The schema to update from.
            merge: Whether to merge or update the grammar.
        """
        self.__schema_builder.add_schema(schema, not merge)
        self.__init_dependencies()

    def to_file(self, path: Path | str = "") -> None:
        """Write the grammar ,schema to a json file.

        Args:
            path: The file path.
                If empty,
                write to a file named after the grammar and with .json extension.
        """
        path = Path(self.name).with_suffix(".json") if path is None else Path(path)
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
    def schema(self) -> DictSchemaType:
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
                if "anyOf" in schema:
                    for sub_schema in schema["anyOf"]:
                        sub_schema["description"] = description
                else:
                    schema["description"] = description
                property_schema.add_schema(schema)

        self.__init_dependencies()

    @classmethod
    def __cast_data_mapping(cls, data: Data) -> DictSchemaType:
        """Cast a data mapping into a JSON-interpretable object.

        Args:
            data: The data mapping.

        Returns:
            The original mapping cast to a JSON-interpretable object.
        """
        _data_dict = dict(data)
        for key, value in data.items():
            _data_dict[key] = cls.__cast_value(value)

        return _data_dict

    @classmethod
    def __cast_value(cls, value: Any):
        """Cast a value to into an JSON-interpretable one.

        Args:
            value: The value.

        Returns:
            The original value cast to a JSON-interpretable object.
        """
        if isinstance(value, complex):
            return value.real

        if isinstance(value, (ndarray, generic)):
            return value.real.tolist()

        if isinstance(value, PathLike):
            return str(value)

        if isinstance(value, Mapping):
            return cls.__cast_data_mapping(value)

        if isinstance(value, Iterable) and not isinstance(value, str):
            return [cls.__cast_value(item) for item in value]

        return value

    def __get_names_to_types(self) -> NamesToTypes:
        """Create the mapping from element names to elements types.

        The elements for which types definitions cannot be expressed as a unique Python
        type, the type is set to ``None``.

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
            if property_json_sub_type not in {"number", "integer", None}:
                LOGGER.warning(
                    self.__WARNING_TEMPLATE,
                    "type",
                    property_json_sub_type,
                    self.name,
                    property_name,
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
                LOGGER.warning(
                    self.__WARNING_TEMPLATE,
                    "feature",
                    feature,
                    self.name,
                    property_name,
                )

    def __init_dependencies(self) -> None:
        """Resets the validator and schema dict."""
        self.__validator = None
        self.__schema = {}

    def _check_name(self, *names: str) -> None:
        self.__schema_builder.check_property_names(*names)

    def __getstate__(self) -> SerializedGrammarType:
        # Ensure self.__schema_builder is filled.
        self.schema  # noqa: B018
        state = dict(self.__dict__)
        # The validator will be recreated on demand.
        del state[f"_{self.__class__.__name__}__validator"]
        # The schema builder cannot be pickled.
        del state[f"_{self.__class__.__name__}__schema_builder"]
        # The defaults cannot be pickled as is because it also depends on the schema
        # builder. So we convert it into a raw dictionary.
        state["defaults"] = dict(state.pop("_defaults"))
        return state

    def __setstate__(
        self,
        state: SerializedGrammarType,
    ) -> None:
        # That will create the missing attributes.
        self.clear()
        self.__dict__.update(state)
        self.__schema_builder.add_schema(
            state[f"_{self.__class__.__name__}__schema"], True
        )
        self._defaults.update(state.pop("defaults"))
