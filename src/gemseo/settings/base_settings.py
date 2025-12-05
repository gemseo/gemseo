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
"""Base settings."""

from __future__ import annotations

from typing import Any
from typing import ClassVar

from pydantic import BaseModel

from gemseo.typing import StrKeyMapping


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

    _TARGET_CLASS_NAME: ClassVar[str] = ""
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
        if cls.__name__.endswith(cls.__SETTINGS_SUFFIX) and (
            # This class directly inherits from BaseSettings
            # and has no target class name.
            not cls._TARGET_CLASS_NAME
            # Or inherits from a class that already has a target class name that
            # should be overriden.
            or len(cls.__mro__) > 2
        ):
            cls._TARGET_CLASS_NAME = cls.__name__.removesuffix(cls.__SETTINGS_SUFFIX)
        fields = cls.__pydantic_fields__
        for name, type_ in cls._INHERITED_FIELD_TYPES.items():
            fields[name].annotation = type_
        for name, default in cls._INHERITED_FIELD_DEFAULTS.items():
            fields[name].default = default
        cls.model_rebuild(force=True)
