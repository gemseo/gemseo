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
"""The type of a variable.

The value of a [VariableType][gemseo.enum.VariableType] is stored in the files a
design space is written to, so a file written by a past release refers to a variable
type by the value that release used. Such a value keeps being accepted, whatever the
release that wrote the file, so that these files remain readable.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final
from warnings import warn

if TYPE_CHECKING:
    from collections.abc import Mapping

legacy_variable_type_values: Final[Mapping[str, str]] = MappingProxyType({
    # Renamed in 7.0.0: the other variable types name a mathematical nature,
    # which "float" did not.
    "float": "real",
})
"""The map from the variable type value of a past release to the current one."""


def warn_legacy_variable_type_value(legacy_value: str, value: str) -> None:
    """Warn that a variable type value of a past release was read.

    Args:
        legacy_value: The variable type value used by a past release.
        value: The current variable type value it was read as.
    """
    warn(
        f"The variable type {legacy_value!r} is deprecated; "
        f"it has been read as {value!r}. "
        "Save the design space again "
        "to store it with the current variable type.",
        DeprecationWarning,
        stacklevel=2,
    )


class VariableType(StrEnum):
    """The type of a variable."""

    CATALOG = "catalog"
    CATEGORICAL = "categorical"
    DISCRETE = "discrete"
    INTEGER = "integer"
    REAL = "real"

    @classmethod
    def _missing_(cls, value: object) -> VariableType | None:
        """Resolve a variable type value used by a past release.

        A file written by a past release refers to a variable type by the value that
        release used, so such a value must keep resolving; the values of the past
        releases are listed in `gemseo.enum._variable_type`.

        Args:
            value: The value that does not name a variable type.

        Returns:
            The variable type the value named in a past release,
            if any, otherwise `None`.
        """
        if not isinstance(value, str):
            return None

        # A CSV file is read as an array of NumPy strings,
        # whose repr would leak into the message.
        legacy_value = str(value)
        new_value = legacy_variable_type_values.get(legacy_value)
        if new_value is None:
            return None

        warn_legacy_variable_type_value(legacy_value, new_value)
        return cls(new_value)

    @classmethod
    def _resolve_value(cls, value: str) -> VariableType | str:
        """Resolve a value naming a variable type, tolerating a value that names none.

        A value used by a past release resolves to the variable type it named then,
        with a `DeprecationWarning`;
        a value naming no variable type at all is returned as it stands,
        so that the caller can report it in an error message of its own.

        Args:
            value: The value to resolve.

        Returns:
            The variable type named by the value, if any, otherwise the value itself.
        """
        try:
            return cls(value)
        except ValueError:
            return value
