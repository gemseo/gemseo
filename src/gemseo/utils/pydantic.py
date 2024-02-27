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
"""Tools for pydantic."""

from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def update_field(
    model: type[BaseModel],
    field_name: str,
    **kwargs: Any,
) -> None:
    """Update a :class:`.Field` of a pydantic model.

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
