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
"""A grammar based on a Pydantic model."""

from __future__ import annotations

import logging
from copy import deepcopy
from enum import Enum
from inspect import isclass
from sys import modules
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import cast
from typing import get_origin

from numpy import dtype as np_dtype
from numpy import ndarray
from pydantic import BaseModel
from pydantic import Strict
from pydantic import ValidationError
from pydantic import create_model
from pydantic.fields import FieldInfo

from gemseo.core.grammars._utils import NOT_IN_THE_GRAMMAR_MESSAGE
from gemseo.core.grammars.base import BaseGrammar
from gemseo.utils.pydantic_ndarray import NDArrayPydantic
from gemseo.utils.pydantic_ndarray import _NDArrayPydantic

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

    from pydantic import ConfigDict
    from typing_extensions import Self

    from gemseo.core.grammars.base import SimpleGrammarTypes
    from gemseo.core.grammars.json_schema import Schema
    from gemseo.typing import StrKeyMapping
    from gemseo.utils.string_tools import MultiLineString

ModelType = type[BaseModel]

LOGGER = logging.getLogger(__name__)

# Pydantic model validation shall be strict to avoid
# situations where for instance a string is cast to an int.
# Nevertheless, fields of type Enum are not strictly validated,
# because the data to be validated may have been processed
# by external tools (e.g., JSON serialization/deserialization)
# and may not match the exact Enum values expected by Pydantic.
# These fields are handled in _copy_model.
_CONFIG_DICT: ConfigDict = {"strict": True}


class PydanticGrammar(BaseGrammar):
    """A grammar based on a Pydantic model.

    When an instance of this class is created from a Pydantic model,
    this model is copied internally to avoid any modifications on it.
    To prevent dangerous validations,
    for instance when validating an int field from a string data,
    the copied model is configured to validate data strictly.
    There is an exception for fields that are of Enum types,
    which are configured to validate data non strictly,
    because this would prevent a safe and natural usage such fields.
    """

    DATA_CONVERTER_CLASS: ClassVar[str] = "PydanticGrammarDataConverter"

    __model: ModelType
    """The Pydantic model."""

    __model_needs_rebuild: bool
    """Whether to rebuild the model before validation when it had runtime changes.

    This is necessary because the Pydantic schema is built at model creation but it does
    not reflect any of the changes done after.
    """

    __SIMPLE_TYPES: ClassVar[set[type]] = {
        _NDArrayPydantic,
        list,
        tuple,
        dict,
        int,
        float,
        complex,
        str,
        bool,
    }
    """The types that can be converted for simple grammars."""

    def __init__(
        self,
        name: str,
        model: ModelType | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: A Pydantic model.
                If `None`, the model will be empty.
            **kwargs: These arguments are not used.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        if model is not None:
            self.__model = _create_model(model)
        # Set the defaults and required names.
        for name, field in self.__model.__pydantic_fields__.items():
            if description := field.description:
                self._descriptions[name] = description

            if field.is_required():
                self._required_names.add(name)
            else:
                self._defaults[name] = field.get_default(call_default_factory=True)

    def __getitem__(self, name: str) -> FieldInfo:
        return self.__model.__pydantic_fields__[name]

    def __len__(self) -> int:
        return len(self.__model.__pydantic_fields__)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__model.__pydantic_fields__)

    def _delitem(self, name: str) -> None:  # noqa:D102
        del self.__model.__pydantic_fields__[name]
        self.__model_needs_rebuild = True

    def _copy(self, grammar: Self) -> None:  # noqa:D102
        grammar.__model = _create_model(self.__model)
        grammar.__model_needs_rebuild = self.__model_needs_rebuild

    def _rename_element(self, current_name: str, new_name: str) -> None:  # noqa:D102
        fields = self.__model.__pydantic_fields__
        fields[new_name] = fields.pop(current_name)
        self.__model_needs_rebuild = True

    def _update(  # noqa:D102
        self,
        grammar: Self,
        excluded_names: Iterable[str],
        merge: bool,
    ) -> None:
        name_to_annotation = {}
        for field_name, field_info in grammar.__model.__pydantic_fields__.items():
            if field_name not in excluded_names:
                name_to_annotation[field_name] = field_info.annotation
        self.__update_from_annotations(name_to_annotation, merge)

    def _update_from_names(  # noqa:D102
        self,
        names: Iterable[str],
        merge: bool,
    ) -> None:
        self.__update_from_annotations(dict.fromkeys(names, NDArrayPydantic), merge)

    def _update_from_data(  # noqa:D102
        self,
        data: StrKeyMapping,
        merge: bool,
    ) -> None:
        name_to_type = {}
        for name, value in data.items():
            if isinstance(value, ndarray):
                name_to_type[name] = _NDArrayPydantic[Any, np_dtype[value.dtype.type]]
            else:
                name_to_type[name] = type(value)
        self._update_from_types(name_to_type, merge=merge)

    def _update_from_types(
        self,
        name_to_type: SimpleGrammarTypes,
        merge: bool,
    ) -> None:
        """Update the grammar from names bound to types.

        For convenience, when a type is exactly `ndarray`,
        it is automatically converted to `NDArrayPydantic`.
        """
        name_to_annotation = dict(name_to_type)
        for name, annotation in name_to_type.items():
            if annotation is ndarray:
                name_to_annotation[name] = NDArrayPydantic
        self.__update_from_annotations(name_to_annotation, merge)

    def update_from_model(self, model: ModelType, merge: bool = False) -> None:
        """Update the grammar from a Pydantic model.

        Unlike [update_from_types()][gemseo.core.grammars.pydantic.PydanticGrammar.update_from_types],
        this method preserves the full field information from the model:
        type annotation, default value or factory, and description.
        Required/optional status is taken from the model's field definitions:
        fields without a default are required.

        Args:
            model: A Pydantic [BaseModel][pydantic.BaseModel] subclass (not an instance).
                If the model has no fields, the grammar is not modified.
            merge: Whether to merge or update the grammar.
        """  # noqa: E501
        if not model.__pydantic_fields__:
            return

        copied_model = _create_model(model)
        fields = copied_model.__pydantic_fields__

        own_fields = self.__model.__pydantic_fields__
        for field_name, field_info in fields.items():
            if merge and field_name in own_fields:
                field_info.annotation = cast(
                    "type[Any]",
                    own_fields[field_name].annotation | field_info.annotation,
                )
            own_fields[field_name] = field_info

            if field_info.is_required():
                self._required_names.add(field_name)
                self._defaults.pop(field_name, None)
            else:
                self._required_names.discard(field_name)
                self._defaults[field_name] = field_info.get_default(
                    call_default_factory=True
                )

            if description := field_info.description:
                self._descriptions[field_name] = description

        self.__model_needs_rebuild = True

    def __update_from_annotations(
        self,
        name_to_annotation: SimpleGrammarTypes,
        merge: bool,
    ) -> None:
        """Update the grammar from names bound to annotations.

        Args:
            name_to_annotation: The mapping from names to annotations.
            merge: Whether to merge or update the grammar.
        """
        fields = self.__model.__pydantic_fields__
        for name, annotation in name_to_annotation.items():
            if merge and name in fields:
                # Pydantic typing for the argument annotation does not handle Union,
                # we cast it.
                annotation = cast("type[Any]", fields[name].annotation | annotation)
            fields[name] = FieldInfo(annotation=annotation)
        self.__model_needs_rebuild = True

    def _clear(self) -> None:  # noqa:D102
        self.__model = _create_model(BaseModel)
        self.__model_needs_rebuild = False
        # The sole purpose of the following attribute is to identify a model created
        # here,
        # and not from an external class deriving from BaseModel,
        self.__model.__internal__ = None  # type: ignore[attr-defined]
        # TODO: This is no longer needed since pydantic 2.10, remove at some point.
        # This is another workaround for pickling a created model.
        self.__model.__pydantic_parent_namespace__ = {}

    def _update_grammar_repr(self, repr_: MultiLineString, properties: Any) -> None:
        repr_.add(f"Type: {properties.annotation}")

    def _validate(  # noqa: D102
        self,
        data: StrKeyMapping,
        error_message: MultiLineString,
    ) -> bool:
        self.__rebuild_model()
        try:
            self.__model.model_validate(data)
        except ValidationError as errors:
            for line in str(errors).split("\n"):
                error_message.add(line)
            return False
        return True

    def _restrict_to(  # noqa:D102
        self,
        names: Iterable[str],
    ) -> None:
        for name in self.keys() - names:
            del self.__model.__pydantic_fields__[name]
            self.__model_needs_rebuild = True

    def _get_name_to_type(self) -> SimpleGrammarTypes:
        """
        Notes:
            For the elements for which types definitions cannot be expressed as a unique
            Python type, the type is set to `None`.
        """  # noqa: D205, D212, D415
        name_to_type = {}
        for name, field in self.__model.__pydantic_fields__.items():
            annotation = field.annotation
            origin = get_origin(annotation)
            pydantic_type = annotation if origin is None else origin
            if pydantic_type not in self.__SIMPLE_TYPES:
                message = (
                    "Unsupported type '%s' in PydanticGrammar '%s' "
                    "for field '%s' in conversion to SimpleGrammar."
                )
                LOGGER.warning(message, origin, self.name, name)
                # This type cannot be converted, use the catch-all type.
                pydantic_type = None

            if pydantic_type is _NDArrayPydantic:
                pydantic_type = ndarray

            name_to_type[name] = pydantic_type

        return name_to_type

    # TODO: API: turn into a getter since it is costly.
    @property
    def schema(self) -> dict[str, Schema]:
        """The dictionary representation of the schema."""
        # The rebuild cannot be postponed for descriptions because these seem to be
        # stored in the schema.
        descriptions = self.descriptions
        for name, field in self.__model.__pydantic_fields__.items():
            if description := descriptions.get(name):
                field.description = description
                self.__model_needs_rebuild = True

        self.__rebuild_model()
        return self.__model.model_json_schema()

    def _check_name(self, *names: str) -> None:
        if not names:
            return

        fields = self.__model.__pydantic_fields__
        for name in names:
            if name not in fields:
                msg = NOT_IN_THE_GRAMMAR_MESSAGE.format(name)
                raise KeyError(msg)

    def __rebuild_model(self) -> None:
        """Rebuild the model if needed."""
        if self.__model_needs_rebuild:
            assert self.__model.model_rebuild(force=True)
            self.__model_needs_rebuild = False

    def __getstate__(self) -> dict[str, Any]:
        self.__rebuild_model()
        state = self.__dict__.copy()
        if hasattr(self.__model, "__internal__"):
            # Workaround for pickling models created at runtime in _clear,
            # because pickling a class requires the source of the class which
            # does not exist in this case.
            # The fields info are pickled instead in order to recreate the model later.
            model_arg_name = f"_{self.__class__.__name__}__model"
            state[model_arg_name] = self.__model.__pydantic_fields__
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if not isinstance(self.__model, type(BaseModel)):
            # Recreate the model from the fields' info.
            fields_info = self.__model
            self._clear()
            fields = self.__model.__pydantic_fields__
            for name, info in fields_info.items():
                fields[name] = info
            self.__model_needs_rebuild = True
            self.__rebuild_model()


def _patch_model(model: ModelType) -> None:
    """Patch the model for internal API change since pydantic 2.10.

    Args:
        model: The model to patch.
    """
    if not hasattr(model, "__pydantic_fields__"):  # pragma: no cover
        model.__pydantic_fields__ = model.model_fields


def _create_model(model: ModelType) -> ModelType:
    """Create a pydantic model by subclassing another one.

    The model validation is made strict but for Enum fields.

    Args:
        model: The model to copy.

    Returns:
        The copied model.
    """
    field_definitions = {}

    if model == BaseModel:
        class_name = "Model"
        # Prefer a neutral name instead of BaseModel
        # which could be misleading.
        schema_title = class_name
        # BaseModel has no __pydantic_fields__,
        # thus the else block is skipped.
    else:
        # Since this class will pretend to be defined in the current module
        # to allow pickling (see below),
        # make sure its name is unique and related to the original model.
        # We create a name similar to a fully qualified name.
        class_name = (model.__module__ + "_" + model.__qualname__).replace(".", "_")
        schema_title = model.__name__

        if not model.__annotations__:
            # Pydantic needs annotations to work properly,
            # via __annotations__,
            # which does not exist when the grammar is instantiated with no model,
            # even if the model has been modified via the grammar API.
            field_definitions = {
                n: (i.annotation, i) for n, i in model.__pydantic_fields__.items()
            }

        # Enum fields shall not be validated strictly.
        for field_name, field_info in model.__pydantic_fields__.items():
            annotation = field_info.annotation
            if isclass(annotation) and issubclass(annotation, Enum):
                field_info_copy = deepcopy(field_info)
                metadata = field_info_copy.metadata

                for item in metadata:
                    if isinstance(item, Strict):
                        item.strict = False
                        break
                else:
                    metadata.append(Strict(strict=False))

                field_definitions[field_name] = (
                    field_info_copy.annotation,
                    field_info_copy,
                )

    derived_model = create_model(
        class_name,
        # title is used when creating the json schema of the model.
        __config__={"title": schema_title, **_CONFIG_DICT},
        # The model copy is made as if it was a derived class from the original model.
        __base__=(model,),
        # Pretend that the copy model is located in the current module,
        # so that it can be pickled properly.
        __module__=_create_model.__module__,
        **field_definitions,
    )

    _patch_model(derived_model)

    if model != BaseModel:
        # Ensure that the class is retrieved when pickling.
        # This is not necessary for BaseModel since the derived class has
        # no specific behavior defined and can be recreated from BaseModel itself.
        setattr(
            modules[_create_model.__module__], derived_model.__name__, derived_model
        )

        # Ensure that FieldInfo metadata are not shared with the original model.
        fields = derived_model.__pydantic_fields__
        for field_name, field_info in fields.items():
            if field_name not in field_definitions:
                fields[field_name] = deepcopy(field_info)

    return derived_model
