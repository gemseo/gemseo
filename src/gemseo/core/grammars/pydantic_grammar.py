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
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Union
from typing import cast

from numpy import ndarray
from pydantic import BaseModel
from pydantic import ValidationError
from pydantic import create_model
from pydantic.fields import FieldInfo
from typing_extensions import Self
from typing_extensions import get_origin

from gemseo.core.grammars._utils import NOT_IN_THE_GRAMMAR_MESSAGE
from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.utils.pydantic_ndarray import NDArrayPydantic
from gemseo.utils.pydantic_ndarray import _NDArrayPydantic

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping

    from gemseo.core.grammars.base_grammar import SimpleGrammarTypes
    from gemseo.core.grammars.json_schema import Schema
    from gemseo.typing import StrKeyMapping
    from gemseo.utils.string_tools import MultiLineString


ModelType = type[BaseModel]

LOGGER = logging.getLogger(__name__)


class PydanticGrammar(BaseGrammar):
    """A grammar based on a Pydantic model.

    The Pydantic model passed to the grammar is used to initialize the grammar defaults.
    Currently, changing the defaults will not update the model.
    Changing the descriptions will update the model when accessing :attr:`.schema`.
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
                If ``None``, the model will be empty.
            **kwargs: These arguments are not used.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        if model is not None:
            self.__model = model
            self.__patch_model(self.__model)
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
        # The deepcopy of a model does not actually deep copies everything,
        # in particular __pydantic_fields__,
        # probably because this is defined at a compiled language level,
        # thus we recreate the model.
        model = self.__model
        grammar.__model = create_model(
            model.__class__.__name__,
            # __config__=model.model_config,
            __doc__=model.__doc__,
            __base__=model.__bases__,
            __module__=model.__module__,
            __validators__=getattr(model, "__validators__", {}),
            **{n: (i.annotation, i) for n, i in model.__pydantic_fields__.items()},
        )
        # The model config cannot be passed to create_model when __base__ is already
        # passed, we set it now.
        grammar.__model.model_config = deepcopy(model.model_config)
        grammar.__model_needs_rebuild = self.__model_needs_rebuild
        self.__patch_model(grammar.__model)

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
        names_to_annotations = {}
        for field_name, field_info in grammar.__model.__pydantic_fields__.items():
            if field_name not in excluded_names:
                names_to_annotations[field_name] = field_info.annotation
        self.__update_from_annotations(names_to_annotations, merge)

    def _update_from_names(  # noqa:D102
        self,
        names: Iterable[str],
        merge: bool,
    ) -> None:
        self.__update_from_annotations(dict.fromkeys(names, NDArrayPydantic), merge)

    def _update_from_types(
        self,
        names_to_types: SimpleGrammarTypes,
        merge: bool,
    ) -> None:
        """Update the grammar from names bound to types.

        For convenience, when a type is exactly ``ndarray``,
        it is automatically converted to ``NDArrayPydantic``.
        """
        names_to_annotations = dict(names_to_types)
        for name, annotation in names_to_types.items():
            if annotation is ndarray:
                names_to_annotations[name] = NDArrayPydantic
        self.__update_from_annotations(names_to_annotations, merge)

    def __update_from_annotations(
        self,
        names_to_annotations: SimpleGrammarTypes,
        merge: bool,
    ) -> None:
        """Update the grammar from names bound to annotations.

        Args:
            names_to_annotations: The mapping from names to annotations.
            merge: Whether to merge or update the grammar.
        """
        fields = self.__model.__pydantic_fields__
        for name, annotation in names_to_annotations.items():
            if merge and name in fields:
                # Pydantic typing for the argument annotation does not handle Union,
                # we cast it.
                annotation = cast(
                    "type[Any]", Union[fields[name].annotation, annotation]
                )
            fields[name] = FieldInfo(annotation=annotation)
        self.__model_needs_rebuild = True

    def _clear(self) -> None:  # noqa:D102
        self.__model = create_model("Model")
        self.__model_needs_rebuild = False
        # The sole purpose of the following attribute is to identify a model created
        # here,
        # and not from an external class deriving from BaseModel,
        self.__model.__internal__ = None  # type: ignore[attr-defined]
        # TODO: This is no longer needed since pydantic 2.10, remove at some point.
        # This is another workaround for pickling a created model.
        self.__model.__pydantic_parent_namespace__ = {}
        self.__patch_model(self.__model)

    def _update_grammar_repr(self, repr_: MultiLineString, properties: Any) -> None:
        repr_.add(f"Type: {properties.annotation}")

    def _validate(  # noqa: D102
        self,
        data: StrKeyMapping,
        error_message: MultiLineString,
    ) -> bool:
        self.__rebuild_model()
        try:
            # The grammars shall be strict on typing and not coerce the data,
            # Pydantic requires a dict, using a mapping fails.
            self.__model.model_validate(dict(data), strict=True)
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

    def _get_names_to_types(self) -> SimpleGrammarTypes:
        """
        Notes:
            For the elements for which types definitions cannot be expressed as a unique
            Python type, the type is set to ``None``.
        """  # noqa: D205, D212, D415
        names_to_types = {}
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

            names_to_types[name] = pydantic_type

        return names_to_types

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

    # TODO: API: remove this deprecated method.
    def set_descriptions(self, descriptions: Mapping[str, str]) -> None:
        """Set the properties descriptions.

        Args:
            descriptions: The mapping from names to the description.
        """
        self._descriptions.update(descriptions)

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

    @staticmethod
    def __patch_model(model: ModelType) -> None:
        """Patch the model for internal API change since pydantic 2.10.

        Args:
            model: The model to patch.
        """
        if not hasattr(model, "__pydantic_fields__"):
            model.__pydantic_fields__ = model.model_fields
