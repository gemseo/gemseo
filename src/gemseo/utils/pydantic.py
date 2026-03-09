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
"""Tools for Pydantic."""

from __future__ import annotations

from functools import partial
from typing import Any
from typing import ClassVar
from typing import TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from gemseo.typing import StrKeyMapping

T = TypeVar("T", bound=BaseModel)


class BaseSettings(
    BaseModel,
    extra="forbid",
    arbitrary_types_allowed=True,
    ser_json_inf_nan="null",
    validate_default=True,
):
    """The base class for settings.

    To change the default values of field defined in base classes,
    set the `_INHERITED_FIELD_DEFAULTS` class attribute in the derived class,
    this attribute maps the field names to the new default field values.
    Changing the field type annotation can be done in a similar way
    with the `_INHERITED_FIELD_TYPES` class attribute.
    """

    __TARGET_CLASS_NAME: ClassVar[str] = ""
    """The name of the class using these settings.

    This name is determined automatically
    by using the naming convention of the setting class like
    `TargetClassName_Settings` where `TargetClassName` is the name
    of the target class that is bound to the settings class.
    """

    _INHERITED_FIELD_DEFAULTS: ClassVar[StrKeyMapping] = {}
    """The mapping from inherited field names to new default values."""

    _INHERITED_FIELD_TYPES: ClassVar[StrKeyMapping] = {}
    """The mapping from inherited field names to new type annotations."""

    __SETTINGS_SUFFIX: ClassVar[str] = "_Settings"
    """The suffix used to determine the target class name."""

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # Do not take into account the class cls and the implicit top base class object.
        base_classes = cls.__mro__[1:-1]
        # Determine automatically the target class name,
        # when the current class name ends with a specific suffix.
        if cls.__name__.endswith(cls.__SETTINGS_SUFFIX) and (
            # And when the target class name is not already set
            # (typically when all the parent classes are not exposed to end users).
            not cls.__TARGET_CLASS_NAME
            # Or when the first parent class it inherits from has a target class name.
            # (typically when the first parent class is exposed to end users)
            # This allows to distinguish the case when the current has the
            # target class name forced in its definition.
            or (
                len(base_classes) > 1
                and cls.__TARGET_CLASS_NAME == base_classes[0].__TARGET_CLASS_NAME
            )
        ):
            cls.__TARGET_CLASS_NAME = cls.__name__.removesuffix(cls.__SETTINGS_SUFFIX)
        fields = cls.__pydantic_fields__
        for name, type_ in cls._INHERITED_FIELD_TYPES.items():
            fields[name].annotation = type_
        for name, default in cls._INHERITED_FIELD_DEFAULTS.items():
            fields[name].default = default
        cls.model_rebuild(force=True)

    @property
    def target_class_name(
        self,
    ) -> str:
        """The name of the class intended to use these settings."""
        return self.__TARGET_CLASS_NAME


def copy_field(name: str, model: type[BaseModel], **kwargs: Any) -> FieldInfo:
    """Copy a Pydantic model `Field`, eventually overridden.

    Args:
        name: The name of the field.
        model: The model to copy the field from.
        **kwargs: The arguments of the field to be overridden.

    Returns:
        The copied field.
    """
    return FieldInfo.merge_field_infos(model.model_fields[name], **kwargs)


def update_field(
    model: type[BaseModel],
    field_name: str,
    **kwargs: Any,
) -> None:
    """Update a [Field][pydantic.Field] of a Pydantic model.

    Args:
        model: The model.
        field_name: The name of the field.
        **kwargs: The arguments of the field to be overridden.
            See `pydantic.Field` for the description of the arguments.
    """
    model.model_fields[field_name] = FieldInfo.merge_field_infos(
        model.model_fields[field_name], **kwargs
    )
    model.model_rebuild(force=True)


def create_model(
    Model: type[T],  # noqa: N803
    settings_model: T | None = None,
    **settings: Any,
) -> T:
    """Create a Pydantic model.

    Args:
        Model: The class of the Pydantic model.
        settings_model: The settings as a Pydantic model.
            If `None`, use `**settings`.
        **settings: The settings.
            These arguments are ignored when `settings_model` is not `None`.

    Returns:
        A Pydantic model

    Raises:
        ValueError: When the class of the `"settings"` argument is not `Model`.
    """
    if settings_model is None:
        return Model(**settings)

    if isinstance(settings_model, Model):
        return settings_model

    msg = (
        f"The Pydantic model must be a {Model.__name__}; "
        f"got {settings_model.__class__.__name__}."
    )
    raise ValueError(msg)


def get_class_name(
    settings_model: BaseSettings | None,
    settings: dict[str, Any],
    class_name_arg: str = "algo_name",
) -> str:
    """Return the name of the class using settings defined as a Pydantic model.

    Args:
        settings_model: The class settings as a Pydantic model.
            If `None`, use `**settings`.
        settings: The settings,
            including the class name (use the keyword `class_name_arg`).
            The function will remove the `class_name_arg` entry.
            These settings are ignored when `settings_model` is not `None`.
        class_name_arg: The name of the argument to set the class name.

    Returns:
        The class name.
    """
    if settings_model is None:
        return settings.pop(class_name_arg)

    return settings_model.target_class_name


# TODO: API: delete when algorithms become classes
get_algo_name = partial(get_class_name, class_name_arg="algo_name")
