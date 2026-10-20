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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A set of functions to handle OS dependent path operations."""

from __future__ import annotations

from pathlib import PurePath
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING
from typing import TypeVar

from numpy import array

from gemseo.util.platform import platform_is_windows

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.util.typing import StringArray

_T = TypeVar("_T")


def to_os_specific(path: Path) -> PureWindowsPath | PurePosixPath:
    """Cast a path to PureWindowsPath on Windows platforms, PurePosixPath otherwise.

    Args:
        path: The path to cast.

    Returns:
        The cast path.
    """
    if platform_is_windows:
        return PureWindowsPath(path)
    return PurePosixPath(path)


def ensure_path_to_array(value: _T) -> _T | StringArray:
    """Convert a value to a one-item string array if it is a path.

    A path is an instance of `pathlib.PurePath` or of one of its subclasses.

    The string does not depend on the platform:
    the parts of the path are separated by forward slashes,
    so that the same path gives the same string on Windows and POSIX platforms.

    Args:
        value: The value to convert.

    Returns:
        The one-item string array of the value if it is a path,
        the value unchanged otherwise.
    """
    if isinstance(value, PurePath):
        return array([value.as_posix()])
    return value
