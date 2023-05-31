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
from copy import copy
from typing import Any
from typing import ClassVar
from typing import Collection
from typing import get_origin
from typing import Iterable
from typing import Iterator
from typing import Mapping
from typing import Sequence
from typing import Type
from typing import Union

from numpy import ndarray
from numpy.typing import NDArray
from pydantic import BaseModel
from pydantic import ValidationError
from pydantic.fields import ModelField

from gemseo.core.discipline_data import MutableData
from gemseo.core.grammars import _pydantic_utils
from gemseo.core.grammars.base_grammar import BaseGrammar
from gemseo.core.grammars.base_grammar import NamesToTypes
from gemseo.core.grammars.json_grammar import DictSchemaType
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.utils.string_tools import MultiLineString

ModelType = Type[BaseModel]

LOGGER = logging.getLogger(__name__)

_pydantic_utils.patch_pydantic()


class PydanticGrammar(BaseGrammar):
    """A grammar based on a pydantic model.

    The pydantic model passed to the grammar is used to initialize the grammar defaults.
    Currently, changing the defaults will not update the model.
    """

    __model: ModelType
    """The pydantic model."""

    __TYPE_TO_PYDANTIC_TYPE: ClassVar[dict[type, type]] = {
        ndarray: NDArray,
    }
    """The mapping from standard types to pydantic specific types."""

    __PYDANTIC_TYPE_TO_SIMPLE_TYPE: ClassVar[dict[type, type]] = {
        ndarray: ndarray,
        list: list,
        tuple: tuple,
        dict: dict,
        int: int,
        float: float,
        str: str,
        bool: bool,
    }
    """The mapping from pydantic types to types for the simple grammar."""

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
        for name, field in self.__model.__fields__.items():
            if not field.required:
                self._defaults[name] = field.get_default()

    def __getitem__(self, name: str) -> ModelField:
        return self.__model.__fields__[name]

    def __len__(self) -> int:
        return len(self.__model.__fields__)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__model.__fields__)

    def _delitem(self, name: str) -> None:  # noqa:D102
        del self.__model.__fields__[name]

    def _copy(self, grammar: PydanticGrammar) -> None:  # noqa:D102
        grammar.__model = copy(self.__model)

    def _rename_element(self, current_name: str, new_name: str) -> None:  # noqa:D102
        fields = self.__model.__fields__
        fields[new_name] = fields.pop(current_name)

    def update(  # noqa:D102
        self,
        grammar: PydanticGrammar,
        exclude_names: Iterable[str] = (),
    ) -> None:
        if not grammar:
            return
        fields = self.__model.__fields__
        for field_name, field in grammar.__model.__fields__.items():
            if field_name in exclude_names:
                continue
            fields[field_name] = copy(field)
        super().update(grammar, exclude_names)

    def update_from_names(  # noqa:D102
        self,
        names: Iterable[str],
    ) -> None:
        if not names:
            return
        model = self.__model
        fields = model.__fields__
        config = model.__config__
        for name in names:
            fields[name] = ModelField(
                name=name,
                type_=NDArray,
                class_validators=None,
                model_config=config,
            )

    def update_from_types(  # noqa:D102
        self,
        names_to_types: NamesToTypes,
        merge: bool = False,
    ) -> None:
        if not names_to_types:
            return
        model = self.__model
        fields = model.__fields__
        config = model.__config__
        for name, type_ in names_to_types.items():
            pydantic_type = self.__TYPE_TO_PYDANTIC_TYPE.get(type_, type_)
            if merge and name in fields:
                field = fields[name]
                field.outer_type_ = Union[field.outer_type_, pydantic_type]
            else:
                fields[name] = ModelField(
                    name=name,
                    type_=pydantic_type,
                    class_validators=None,
                    model_config=config,
                )

    def _clear(self) -> None:  # noqa:D102
        class Model(BaseModel):  # noqa: D102
            pass

        self.__model = Model

    def _repr_required_elements(self, text: MultiLineString) -> None:  # noqa: D102
        for name, field in self.__model.__fields__.items():
            if field.required:
                text.add(f"{name}: {field.outer_type_}")

    def _repr_optional_elements(self, text: MultiLineString) -> None:  # noqa: D102
        for name, field in self.__model.__fields__.items():
            if not field.required:
                text.add(f"{name}: {field.outer_type_}")
                if name in self._defaults:
                    text.indent()
                    text.add(f"default: {self._defaults[name]}")
                    text.dedent()

    def _validate(  # noqa: D102
        self,
        data: MutableData,
        error_message: MultiLineString,
    ) -> bool:
        try:
            self.__model.parse_obj(data)
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
        type_ = get_origin(self.__model.__fields__[name].outer_type_)
        if type_ is None:
            return False
        if numeric_only:
            return issubclass(type_, ndarray)
        return issubclass(type_, Collection)

    def _restrict_to(  # noqa:D102
        self,
        names: Sequence[str],
    ) -> None:
        for name in self.keys() - names:
            del self.__model.__fields__[name]

    def to_simple_grammar(self) -> SimpleGrammar:
        """
        Notes:
            For the elements for which types definitions cannot be expressed as a unique
            Python type, the type is set to ``None``.
        """  # noqa: D205, D212, D415
        names_to_types = {}
        for name, field in self.__model.__fields__.items():
            outer_type_ = field.outer_type_
            origin = get_origin(outer_type_)
            pydantic_type = outer_type_ if origin is None else origin
            simple_type = self.__PYDANTIC_TYPE_TO_SIMPLE_TYPE.get(pydantic_type)
            if simple_type is None:
                message = (
                    "Unsupported type '%s' in PydanticGrammar '%s' "
                    "for field '%s' in conversion to SimpleGrammar."
                )
                LOGGER.warning(message, origin, self.name, name)
            names_to_types[name] = simple_type

        return SimpleGrammar(
            self.name,
            names_to_types=names_to_types,
            required_names=self.required_names,
        )

    @property
    def required_names(self) -> set[str]:  # noqa:D102
        return {
            name for name, field in self.__model.__fields__.items() if field.required
        }

    @property
    def schema(self) -> dict[str, DictSchemaType]:
        """The dictionary representation of the schema."""
        return self.__model.schema()

    # TODO: keep for backward compatibility but remove at some point since
    # the descriptions are set in the model.
    def set_descriptions(self, descriptions: Mapping[str, str]) -> None:
        """Set the properties descriptions.

        Args:
            descriptions: The mapping from names to the description.
        """
        if not descriptions:
            return

        for name, field in self.__model.__fields__.items():
            description = descriptions.get(name)
            if description:
                field.field_info.description = description

    def _check_name(self, *names: str) -> None:
        fields = self.__model.__fields__
        for name in names:
            if name not in fields:
                raise KeyError(f"The name {name} is not in the grammar.")
