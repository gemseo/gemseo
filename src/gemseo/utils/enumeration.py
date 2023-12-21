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
#        :author:  Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Enumerations tools."""

from __future__ import annotations

import inspect
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import Enum


def merge_enums(
    name: str,
    base_enum_class: type[Enum],
    *enums: type[Enum],
    doc: str = "",
) -> type[Enum]:
    """Create an enum from other ones.

    This is useful because an enum class cannot be derived for extension with other
    enum items.

    Args:
        name: The name of the enum class to create.
        base_enum_class: The base enum class to derive from.
        *enums: The enum classes to be merged in.
        doc: The new enum class docstring.

    Returns:
        The created enum class.
    """
    # We need to determine where the new enum is created from because it cannot be
    # done otherwise and pickling fails.
    caller_f_locals = inspect.stack()[1][0].f_locals
    new_enum = base_enum_class(
        name,
        chain(*(e.__members__.items() for e in enums)),
        module=caller_f_locals["__module__"],
        qualname=caller_f_locals["__qualname__"] + f".{name}",
    )

    if doc:
        # Set the docstrings for the class.
        new_enum.__doc__ = doc

    # Set the docstrings for the members by copying the ones from the parent enums
    # members.
    new_members = new_enum.__members__
    for enum in chain(*enums):
        new_members[enum.name].__doc__ = enum.__doc__

    return new_enum
