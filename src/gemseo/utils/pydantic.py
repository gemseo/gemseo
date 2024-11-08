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

from typing import Any
from typing import TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo

T = TypeVar("T", bound=BaseModel)


def copy_field(name: str, model: type[BaseModel], **kwargs: Any) -> FieldInfo:
    """Copy a Pydantic model ``Field``, eventually overriden.

    Args:
        name: The name of the field.
        model: The model to copy the field from.
        **kwargs: The arguments of the field to be overriden.

    Returns:
        The copied field.
    """
    return FieldInfo.merge_field_infos(model.model_fields[name], **kwargs)


def update_field(
    model: type[BaseModel],
    field_name: str,
    **kwargs: Any,
) -> None:
    """Update a :class:`.Field` of a Pydantic model.

    Args:
        model: The model.
        field_name: The name of the field.
        **kwargs: The arguments of the field to be overridden.
            See :func:`.Field` for the description of the arguments.
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
            If ``None``, use ``**settings``.
        **settings: The settings.
            These arguments are ignored when ``settings_model`` is not ``None``.

    Returns:
        A Pydantic model

    Raises:
        ValueError: When the class of the ``"settings"`` argument is not ``Model``.
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


def get_algo_name(settings_model: BaseModel | None, settings: dict[str, Any]) -> str:
    """Return the algorithm name.

    Args:
        settings_model: The algorithm settings as a Pydantic model.
            If ``None``, use ``**settings``.
        settings: The algorithm settings,
            including the algorithm name (use the keyword ``"algo_name"``).
            The function will remove the ``"algo_name"`` entry.
            These settings are ignored when ``settings_model`` is not ``None``.

    Returns:
        The algorithm name.
    """
    if settings_model is None:
        return settings.pop("algo_name")

    return settings_model._TARGET_CLASS_NAME
