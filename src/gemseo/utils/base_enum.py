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
"""Custom Enumerations.

The `.BaseEnum` class enables the specification of options which were previously defined
using class attributes. It enables backward compatibility, as the user can either
provide the Enum member or its name as a string. The conversion is then made implicitly.
"""
from __future__ import annotations

from enum import Enum
from enum import EnumMeta


class MetaEnum(EnumMeta):
    """An Enum meta-class to subclass the `in` behavior."""

    def __contains__(cls, item) -> bool:  # noqa: N805
        if isinstance(item, cls):
            return super().__contains__(item)
        elif isinstance(item, str):
            return item in cls.__members__.keys()
        return False


class BaseEnum(Enum, metaclass=MetaEnum):
    """A base Enum class that can be compared to strings."""

    @classmethod
    def get_member_from_name(
        cls,
        value: str | BaseEnum,
    ) -> BaseEnum:
        """Return an Enum member from a name or a member.

        This class method returns an Enum member
        either from the name of the member or the member itself.
        It enables backward compatibility with the use of class attributes
        for options.

        Args:
            value: An Enum member name or an Enum member.

        Raises:
            TypeError: If the Enum member is not from the same Enum class.
        """
        if isinstance(value, str):
            return cls.__members__[value]
        elif isinstance(value, cls):
            return value

        raise TypeError(
            "The type of value is {} but {} or str are expected.".format(
                type(value), str(cls)
            )
        )

    def __eq__(
        self,
        other: BaseEnum | str,
    ) -> bool:
        if isinstance(other, self.__class__):
            return other.value == self.value
        elif isinstance(other, str):
            return other == self.name
        return False

    def __str__(self) -> str:
        return self.name


class CamelCaseEnum(Enum):
    """Enum that are represented as the camel case of the key name."""

    def __str__(self) -> str:
        return self.name.title().replace("_", "")
