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
from contextlib import contextmanager
from copy import deepcopy
from numbers import Complex
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import cast

from fastjsonschema import JsonSchemaException
from fastjsonschema import compile as compile_schema
from numpy import ndarray

from gemseo.core.grammars._python_to_json import PYTHON_TO_JSON_TYPES
from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.core.grammars.json_schema import MutableMappingSchemaBuilder
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from typing_extensions import Self

    from gemseo.core.grammars.base_grammar import SimpleGrammarTypes
    from gemseo.core.grammars.json_schema import Properties
    from gemseo.core.grammars.json_schema import Schema
    from gemseo.typing import StrKeyMapping
    from gemseo.utils.string_tools import MultiLineString

LOGGER = logging.getLogger(__name__)


class JSONGrammar(BaseGrammar):
    """A grammar based on a JSON schema.

    For the dictionary-like methods similar to ``update``,
    when a key exists in both grammars,
    the values can be merged instead of being
    updated by passing ``merge = True``.
    In that case, the resulting grammar will allow any of the values.

    When using :meth:`.update_from_types`,
    it is assumed that a grammar element of type ``ndarray`` is a number.
    """

    DATA_CONVERTER_CLASS: ClassVar[str] = "JSONGrammarDataConverter"

    __validator: Callable[[StrKeyMapping], None] | None
    """The schema validator."""

    __schema: Schema
    """The schema stored as a dictionary."""

    __schema_builder: MutableMappingSchemaBuilder
    """The internal schema object.

    This object has its own handling of the required names,
    but it is almost not used since this handling is done in the base class
    :class:`.BaseGrammar`.
    Nevertheless, we use the required names that this object can read from
    external sources (json schema from a dict or a file).
    We also need to populate the required names of this object when it is
    used to produce a json schema (to a file or a dict).
    Except from those 2 uses, we try to keep the required names of this object
    empty to avoid any side effects with the ones from the base class.
    """

    __JSON_TO_PYTHON_TYPES: Final[dict[str, type]] = {
        "array": ndarray,
        "string": str,
        "integer": int,
        "boolean": bool,
        # As opposed to the complex type, Complex follows sub-typing,
        # such that a float number is a subtype of complex number.
        # This is especially important when converting to SimpleGrammar.
        "number": Complex,
    }
    """The mapping from JSON types to Python types."""

    __PYTHON_TO_JSON_TYPES: Final[dict[type, str]] = PYTHON_TO_JSON_TYPES
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
        descriptions: Mapping[str, str] = READ_ONLY_EMPTY_DICT,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            file_path: The path to a JSON schema file.
                If empty, do not initialize the grammar from a JSON schema file.
            descriptions: The descriptions of the elements read from ``file_path``,
                in the form: ``{element_name: element_meaning}``.
                If empty, use the descriptions available in the JSON schema if any.
            **kwargs: These arguments are not used.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        if file_path:
            self.update_from_file(file_path)
            self._descriptions.update(descriptions)

    def __getitem__(self, name: str) -> Any:
        return self.__schema_builder[name]

    def __len__(self) -> int:
        return len(self.__schema_builder)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.__schema_builder)

    def _delitem(self, name: str) -> None:  # noqa:D102
        del self.__schema_builder[name]
        self.__init_dependencies()

    def _copy(self, grammar: Self) -> None:
        # Updating is much faster than deep copying a schema builder.
        grammar.__schema_builder.add_schema(self.__schema_builder, True)

    def _rename_element(self, current_name: str, new_name: str) -> None:  # noqa: D102
        self.__schema_builder.properties[new_name] = (
            self.__schema_builder.properties.pop(current_name)
        )
        self.__init_dependencies()

    def _update(
        self,
        grammar: Self,
        excluded_names: Iterable[str],
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

        if excluded_names:
            schema_builder = deepcopy(grammar.__schema_builder)
            for name in excluded_names:
                if name in schema_builder:
                    del schema_builder[name]
        else:
            schema_builder = grammar.__schema_builder

        self.__schema_builder.add_schema(schema_builder, not merge)
        self.__init_dependencies()

    def _update_from_names(
        self,
        names: Iterable[str],
        merge: bool,
    ) -> None:
        for name in names:
            self.__schema_builder.add_object({name: [0.0]}, not merge)
        self.__schema_builder.required.clear()
        self.__init_dependencies()

    def _update_from_data(
        self,
        data: StrKeyMapping,
        merge: bool,
    ) -> None:
        """
        Notes:
            The types of the values of the ``data`` will be converted
            to JSON Schema types and define the elements of the JSON Schema.
        """  # noqa: D205, D212, D415
        self.__schema_builder.add_object(self.__cast_data_mapping(data), not merge)
        self.__schema_builder.required.clear()
        self.__init_dependencies()

    def _update_from_types(  # noqa: D102
        self,
        names_to_types: SimpleGrammarTypes,
        merge: bool,
    ) -> None:
        try:
            properties: Properties = {}
            for element_name, element_type in names_to_types.items():
                if element_type is None:
                    sub_property = {}
                else:
                    json_type = self.__PYTHON_TO_JSON_TYPES[element_type]
                    sub_property = {"type": json_type}
                    if element_type == ndarray:
                        sub_property["items"] = {"type": "number"}
                properties[element_name] = sub_property
        # TODO: API: use TypeError.
        except KeyError as error:
            msg = f"Unsupported python type for a JSON Grammar: {error}"
            raise KeyError(msg) from None

        schema: Schema = {
            "$schema": "http://json-schema.org/draft-04/schema",
            "type": "object",
            "properties": properties,
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
        data: StrKeyMapping,
        error_message: MultiLineString,
    ) -> bool:
        if self.__validator is None:
            self._create_validator()
            assert self.__validator is not None

        try:
            self.__validator(self.__cast_data_mapping(data))
        except JsonSchemaException as error:
            if not error.args[0].startswith("data must contain"):
                error_message.add(f"error: {error.args[0]}")
            return False

        return True

    def _restrict_to(  # noqa: D102
        self,
        names: Iterable[str],
    ) -> None:
        for element_name in self.__schema_builder.keys() - names:
            del self.__schema_builder[element_name]
        self.__init_dependencies()

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
            msg = f"Cannot update the grammar from non existing file: {path}."
            raise FileNotFoundError(msg)
        self.update_from_schema(json.loads(path.read_text()), merge)

    def update_from_schema(self, schema: Schema, merge: bool = False) -> None:
        """Update the grammar from a json schema.

        Args:
            schema: The schema to update from.
            merge: Whether to merge or update the grammar.
        """
        self.__schema_builder.add_schema(schema, not merge)
        self.__init_dependencies()
        self._required_names |= self.__schema_builder.required
        self.__schema_builder.required.clear()
        for (
            property_name,
            property_schema,
        ) in self.__schema_builder.properties.items():
            schema = property_schema.to_schema()
            schemas = schema.get("anyOf", (schema,))
            for _schema in schemas:
                if description := _schema.get("description"):
                    # We use the first description.
                    self._descriptions[property_name] = description
                    break

    def to_file(self, path: Path | str = "") -> None:
        """Write the grammar ,schema to a json file.

        Args:
            path: The file path.
                If empty,
                write to a file named after the grammar and with .json extension.
        """
        path = Path(self.name).with_suffix(".json") if not path else Path(path)
        with self.__sync_required_names():
            path.write_text(json.dumps(self.schema, indent=2), encoding="utf-8")

    def to_json(self, *args: Any, **kwargs: Any) -> str:
        """Return the JSON representation of the grammar schema.

        Args:
            *args: The positional arguments passed to :func:`json.dumps`.
            **kwargs: The keyword arguments passed to :func:`json.dumps`.

        Returns:
            The JSON representation of the schema.
        """
        with self.__sync_required_names():
            return cast("str", json.dumps(self.schema, *args, **kwargs))

    @contextmanager
    def __sync_required_names(self) -> Iterator[None]:
        """Synchronize the required names while processing the schema builder."""
        self.__schema_builder.required.update(self._required_names)
        yield
        self.__schema_builder.required.clear()

    @property
    def schema(self) -> Schema:
        """The dictionary representation of the schema."""
        if not self.__schema:
            with self.__sync_required_names():
                self.__schema = self.__schema_builder.to_schema()

            descriptions = self._descriptions
            for (
                property_name,
                property_schema,
            ) in self.__schema.get("properties", {}).items():
                if description := descriptions.get(property_name):
                    schemas = property_schema.get("anyOf", (property_schema,))
                    for schema in schemas:
                        schema["description"] = description

        return self.__schema

    def _create_validator(self) -> None:
        """Create the schema validator."""
        self.schema.pop("id", None)
        self.schema.pop("required", None)
        self.__validator = compile_schema(self.schema)

    @BaseGrammar.descriptions.setter
    def descriptions(self, data: StrKeyMapping) -> None:  # noqa: D102
        BaseGrammar.descriptions.fset(self, data)
        self.__init_dependencies()

    # TODO: API: remove this deprecated method.
    def set_descriptions(self, descriptions: Mapping[str, str]) -> None:
        """Set the properties descriptions.

        Args:
            descriptions: The descriptions, mapping properties names
                to the description.
        """
        if not descriptions:
            return

        self._descriptions.update(descriptions)
        self.__init_dependencies()

    @classmethod
    def __cast_data_mapping(cls, data: StrKeyMapping) -> dict[str, Any]:
        """Cast a data mapping into a JSON-interpretable object.

        Args:
            data: The data mapping.

        Returns:
            The original mapping cast to a JSON-interpretable object.
        """
        data_dict = dict(data)
        for key, value in data.items():
            data_dict[key] = cls.__cast_value(value)

        return data_dict

    @classmethod
    def __cast_value(cls, value: Any) -> Any:
        """Cast a value to into an JSON-interpretable one.

        Args:
            value: The value.

        Returns:
            The original value cast to a JSON-interpretable object.
        """
        if isinstance(value, complex):
            return value.real

        if isinstance(value, ndarray):
            return value.real.tolist()

        if isinstance(value, PathLike):
            return str(value)

        if isinstance(value, Mapping):
            return cls.__cast_data_mapping(value)

        if isinstance(value, Iterable) and not isinstance(value, str):
            return [cls.__cast_value(item) for item in value]

        return value

    def _get_names_to_types(self) -> SimpleGrammarTypes:
        names_to_types = {}

        for property_name, property_description in self.schema.get(
            "properties"
        ).items():
            if property_description["type"] not in self.__JSON_TO_PYTHON_TYPES:
                property_type = None
            else:
                property_type = self.__JSON_TO_PYTHON_TYPES[
                    property_description["type"]
                ]
            names_to_types[property_name] = property_type

        return names_to_types

    def __init_dependencies(self) -> None:
        """Resets the validator and schema dict."""
        self.__validator = None
        self.__schema = {}

    def _check_name(self, *names: str) -> None:
        self.__schema_builder.check_property_names(*names)

    def __getstate__(self) -> dict[str, Any]:
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
        state: dict[str, Any],
    ) -> None:
        # That will create the missing attributes.
        self.clear()
        self.__dict__.update(state)
        self.__schema_builder.add_schema(
            state[f"_{self.__class__.__name__}__schema"], True
        )
        self._defaults.update(cast("StrKeyMapping", state.pop("defaults")))
