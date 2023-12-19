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
"""A grammar based on a pydantic model."""

from __future__ import annotations

import logging
from collections.abc import Collection
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Union

from numpy import ndarray
from pydantic import BaseModel
from pydantic import ValidationError
from pydantic import create_model
from pydantic.fields import FieldInfo
from typing_extensions import get_origin

from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.core.grammars.base_grammar import NamesToTypes
from gemseo.core.grammars.pydantic_ndarray import NDArrayPydantic
from gemseo.core.grammars.pydantic_ndarray import _NDArrayPydantic
from gemseo.core.grammars.simple_grammar import SimpleGrammar

if TYPE_CHECKING:
    from gemseo.core.discipline_data import MutableData
    from gemseo.core.grammars.json_grammar import DictSchemaType
    from gemseo.utils.string_tools import MultiLineString

ModelType = type[BaseModel]

LOGGER = logging.getLogger(__name__)


class PydanticGrammar(BaseGrammar):
    """A grammar based on a pydantic model.

    The pydantic model passed to the grammar is used to initialize the grammar defaults.
    Currently, changing the defaults will not update the model.
    """

    DATA_CONVERTER_CLASS: ClassVar[str] = "PydanticGrammarDataConverter"

    __model: ModelType
    """The pydantic model."""

    __model_needs_rebuild: bool
    """Whether to rebuild the model before validation when it had runtime changes.

    This is necessary because the pydantic schema is built at model creation but it does
    not reflect any of the changes done after.
    """

    __SIMPLE_TYPES: ClassVar[tuple[type]] = {
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
    ):
        """
        Args:
            model: A pydantic model.
                If ``None``, the model will be empty.
            **kwargs: These arguments are not used.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        if model is not None:
            self.__model = model
        # Set the defaults.
        for name, field in self.__model.model_fields.items():
            if not field.is_required():
                self._defaults[name] = field.get_default(call_default_factory=True)

    def __getitem__(self, name: str) -> FieldInfo:
        return self.__model.model_fields[name]

    def __len__(self) -> int:
        return len(self.__model.model_fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__model.model_fields)

    def _delitem(self, name: str) -> None:  # noqa:D102
        del self.__model.model_fields[name]
        self.__model_needs_rebuild = True

    def _copy(self, grammar: PydanticGrammar) -> None:  # noqa:D102
        grammar.__model = copy(self.__model)
        grammar.__model_needs_rebuild = self.__model_needs_rebuild

    def _rename_element(self, current_name: str, new_name: str) -> None:  # noqa:D102
        fields = self.__model.model_fields
        fields[new_name] = fields.pop(current_name)
        self.__model_needs_rebuild = True

    def update(  # noqa:D102
        self,
        grammar: PydanticGrammar,
        exclude_names: Iterable[str] = (),
    ) -> None:
        if not grammar:
            return
        fields = self.__model.model_fields
        for field_name, field in grammar.__model.model_fields.items():
            if field_name in exclude_names:
                continue
            fields[field_name] = copy(field)
            self.__model_needs_rebuild = True
        super().update(grammar, exclude_names)

    def update_from_names(  # noqa:D102
        self,
        names: Iterable[str],
    ) -> None:
        if not names:
            return
        fields = self.__model.model_fields
        for name in names:
            fields[name] = FieldInfo(name=name, annotation=NDArrayPydantic)
        self.__model_needs_rebuild = True

    def update_from_types(  # noqa:D102
        self,
        names_to_types: NamesToTypes,
        merge: bool = False,
    ) -> None:
        """Update the grammar from names bound to types.

        The updated elements are required.
        For convenience, when a type is exactly ``ndarray``,
        it is automatically converted to ``NDArrayPydantic``.
        """
        if not names_to_types:
            return
        fields = self.__model.model_fields
        for name, annotation in names_to_types.items():
            if annotation is ndarray:
                annotation = _NDArrayPydantic
            if merge and name in fields:
                annotation = Union[fields[name].annotation, annotation]
            fields[name] = FieldInfo(name=name, annotation=annotation)
        self.__model_needs_rebuild = True

    def _clear(self) -> None:  # noqa:D102
        self.__model = create_model("Model")
        self.__model_needs_rebuild = False

    def _update_grammar_repr(self, repr_: MultiLineString, properties: Any) -> None:
        repr_.add(f"Type: {properties.annotation}")

    def _validate(  # noqa: D102
        self,
        data: MutableData,
        error_message: MultiLineString,
    ) -> bool:
        self.__rebuild_model()
        try:
            # The grammars shall be strict on typing and not coerce the data.
            self.__model.model_validate(data, strict=True)
        except ValidationError as errors:
            for line in str(errors).split("\n"):
                error_message.add(line)
            return False
        return True

    def is_array(  # noqa:D102
        self,
        name: str,
        numeric_only: bool = False,
    ) -> bool:
        self._check_name(name)
        if numeric_only:
            return self.data_converter.is_numeric(name)
        annotation = self.__model.model_fields[name].annotation
        type_origin = get_origin(annotation)
        if type_origin is None:
            # This is a container with no information on the type of its contents.
            # This is the case of a type just declared as ndarray for instance.
            return False
        return issubclass(type_origin, Collection)

    def _restrict_to(  # noqa:D102
        self,
        names: Sequence[str],
    ) -> None:
        for name in self.keys() - names:
            del self.__model.model_fields[name]
            self.__model_needs_rebuild = True

    def to_simple_grammar(self) -> SimpleGrammar:
        """
        Notes:
            For the elements for which types definitions cannot be expressed as a unique
            Python type, the type is set to ``None``.
        """  # noqa: D205, D212, D415
        names_to_types = {}
        for name, field in self.__model.model_fields.items():
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

        return SimpleGrammar(
            self.name,
            names_to_types=names_to_types,
            required_names=self.required_names,
        )

    @property
    def required_names(self) -> set[str]:  # noqa:D102
        return {
            name
            for name, field in self.__model.model_fields.items()
            if field.is_required()
        }

    @property
    def schema(self) -> dict[str, DictSchemaType]:
        """The dictionary representation of the schema."""
        return self.__model.model_json_schema()

    # TODO: keep for backward compatibility but remove at some point since
    # the descriptions are set in the model.
    def set_descriptions(self, descriptions: Mapping[str, str]) -> None:
        """Set the properties descriptions.

        Args:
            descriptions: The mapping from names to the description.
        """
        if not descriptions:
            return

        # The rebuild cannot be postponed for descriptions because these seem to be
        # stored in the schema.
        for name, field in self.__model.model_fields.items():
            description = descriptions.get(name)
            if description:
                field.description = description
                self.__model_needs_rebuild = True

        self.__rebuild_model()

    def _check_name(self, *names: str) -> None:
        fields = self.__model.model_fields
        for name in names:
            if name not in fields:
                raise KeyError(f"The name {name} is not in the grammar.")

    def __rebuild_model(self) -> None:
        """Rebuild the model if needed."""
        if self.__model_needs_rebuild:
            self.__model.model_rebuild(force=True)
            self.__model_needs_rebuild = False
