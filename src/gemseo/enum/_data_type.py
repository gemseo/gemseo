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
"""The type of variable data.

The value of a [DataType][gemseo.enum.DesignVariableType] is stored in the files a
design space is written to, so a file written by a past release refers to a data type
by the value that release used. Such a value keeps being accepted, whatever the
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

legacy_data_type_values: Final[Mapping[str, str]] = MappingProxyType({
    # Renamed in 7.0.0: the other data types name a mathematical nature,
    # which "float" did not.
    "float": "real",
})
"""The map from the data type value of a past release to the current one."""


def warn_legacy_data_type_value(legacy_value: str, value: str) -> None:
    """Warn that a data type value of a past release was read.

    Args:
        legacy_value: The data type value used by a past release.
        value: The current data type value it was read as.
    """
    warn(
        f"The variable data type {legacy_value!r} is deprecated; "
        f"it has been read as {value!r}. "
        "Save the design space again "
        "to store it with the current data type.",
        DeprecationWarning,
        stacklevel=2,
    )


class DataType(StrEnum):
    """The type of variable data."""

    DISCRETE = "discrete"
    INTEGER = "integer"
    REAL = "real"

    @classmethod
    def _missing_(cls, value: object) -> DataType | None:
        """Resolve a data type value used by a past release.

        A file written by a past release refers to a data type by the value that
        release used, so such a value must keep resolving; the values of the past
        releases are listed in `gemseo.enum._data_type`.

        Args:
            value: The value that does not name a data type.

        Returns:
            The data type the value named in a past release,
            if any, otherwise `None`.
        """
        if not isinstance(value, str):
            return None

        # A CSV file is read as an array of NumPy strings,
        # whose repr would leak into the message.
        legacy_value = str(value)
        new_value = legacy_data_type_values.get(legacy_value)
        if new_value is None:
            return None

        warn_legacy_data_type_value(legacy_value, new_value)
        return cls(new_value)

    @classmethod
    def _resolve_value(cls, value: str) -> DataType | str:
        """Resolve a value naming a data type, tolerating a value that names none.

        A value used by a past release resolves to the data type it named then,
        with a `DeprecationWarning`;
        a value naming no data type at all is returned as it stands,
        so that the caller can report it in an error message of its own.

        Args:
            value: The value to resolve.

        Returns:
            The data type named by the value, if any, otherwise the value itself.
        """
        try:
            return cls(value)
        except ValueError:
            return value
