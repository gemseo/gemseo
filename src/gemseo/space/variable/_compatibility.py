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
"""Compatibility with the data type values of the past releases.

The value of a [DataType][gemseo.space.variable.base.DataType] is stored in the
files a design space is written to, so a file written by a past release refers to
a data type by the value that release used. Such a value keeps being accepted,
whatever the release that wrote the file, so that these files remain readable.
"""

from __future__ import annotations

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
