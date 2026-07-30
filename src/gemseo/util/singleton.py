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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Implementations of the singleton design pattern."""

from __future__ import annotations

from os.path import realpath
from typing import Any
from typing import ClassVar


def _check_args(name: str, args: tuple[Any, ...]) -> None:
    """Check whether there is at least one argument.

    Args:
        name: The multiton class name.
        args: The arguments, if any.

    Raises:
        ValueError: If the constructor has no argument.
    """
    if not args:
        msg = f"{name} subclasses need at least one argument in the constructor."
        raise ValueError(msg)


class SingleInstancePerAttributeId(type):
    """A multiton that depends on the object's `id` passed.

    A subclass is instantiated
    only if the `id` of the object provided as the first constructor argument
    is different from the `id` of every object previously provided as that argument.
    """

    instances: ClassVar[dict[tuple[int, int], Any]] = {}

    # Eclipse is not happy with "cls" as first
    # argument but this is an eclipse bug.
    # function.DisciplineAdapterGenerator should have self as first parameter"
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa:D102
        _check_args("SingleInstancePerAttributeId", args)
        # id = memory address of the object, which is unique
        instance_key = (id(cls), id(args[0]))
        instance = cls.instances.get(instance_key)
        if instance is None:
            instance = type.__call__(cls, *args, **kwargs)
            cls.instances[instance_key] = instance
        return instance


class SingleInstancePerFileAttribute(type):
    """A multiton that depends on the object's value passed.

    A subclass is instantiated
    only if the value of the object provided as the first constructor argument
    does not equal the value from every object previously provided as that argument.
    """

    instances: ClassVar[dict[tuple[int, str], Any]] = {}

    # Eclipse is not happy with "cls" as first
    # argument but this is an eclipse bug.
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:  # noqa:D102
        _check_args("SingleInstancePerFileAttribute", args)

        file_path = args[0]
        if not isinstance(file_path, str):
            msg = f"Argument 0 is not a string but of type {type(file_path)}."
            raise TypeError(msg)

        file_path = realpath(file_path)
        instance_key = (id(cls), file_path)
        instance = cls.instances.get(instance_key)
        if instance is None:
            instance = type.__call__(cls, *args, **kwargs)
            cls.instances[instance_key] = instance
        return instance
