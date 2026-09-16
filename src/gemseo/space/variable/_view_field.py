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
"""A descriptor reading a field that holds a frozen array as a view of it."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import ndarray

if TYPE_CHECKING:
    from typing import Any

    from pydantic import BaseModel


class ArrayViewField:
    """A data descriptor reading a field that holds a frozen array as a view of it.

    A frozen array cannot be mutated in place,
    but NumPy lets a caller reassign the shape, the strides and the data type
    of a read-only array, and these belong to the array object itself;
    so the owner of a frozen array hands out a view of the array it stores,
    and a reassignment reaches the view of the caller only.

    Pydantic keeps the values of the fields of a model in the `__dict__` of the instance
    and a property cannot take the name of a field,
    so this descriptor is set on a model class once pydantic has built it,
    with
    [expose_fields_as_views][gemseo.space.variable._view_field.expose_fields_as_views],
    and reads the value of the field from that `__dict__`.
    Read from the class itself, it raises an `AttributeError`,
    as a field does once pydantic has collected it,
    so that pydantic does not mistake it for the default of the field
    when building a subclass, which inherits the field unchanged.
    """

    __slots__ = ("_name",)

    _name: str
    """The name of the field."""

    def __init__(self, name: str) -> None:
        """
        Args:
            name: The name of the field.
        """  # noqa: D205, D212
        self._name = name

    def __get__(self, instance: BaseModel | None, owner: type | None = None) -> Any:
        if instance is None:
            raise AttributeError(self._name)

        try:
            value = instance.__dict__[self._name]
        except KeyError:
            raise AttributeError(self._name) from None

        # Until a validator converts and freezes it,
        # the value is whatever the caller supplied.
        return value.view() if isinstance(value, ndarray) else value

    def __set__(self, instance: BaseModel, value: Any) -> None:
        # A frozen model refuses the assignment of a field before reaching here;
        # this method makes the descriptor a data one,
        # which takes precedence over the __dict__ of the instance.
        msg = f"{self._name} is read-only."
        raise AttributeError(msg)


def expose_fields_as_views(cls: type[BaseModel], *names: str) -> None:
    """Read the fields of a model that hold frozen arrays as views of them.

    Args:
        cls: The model class, once pydantic has built it.
        *names: The names of the fields.

    Raises:
        ValueError: If a name is not the one of a field of the model.
    """
    for name in names:
        if name not in cls.model_fields:
            msg = f"{name} is not a field of {cls.__name__}."
            raise ValueError(msg)

        setattr(cls, name, ArrayViewField(name))
