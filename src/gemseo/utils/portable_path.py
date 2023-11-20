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

from pathlib import Path
from pathlib import PurePosixPath
from pathlib import PureWindowsPath

from gemseo.utils.platform import PLATFORM_IS_WINDOWS


def to_os_specific(path: Path) -> PureWindowsPath | PurePosixPath:
    """Cast a path to PureWindowsPath on Windows platforms, PurePosixPath otherwise.

    Args:
        path: The path to cast.

    Returns:
        The cast path.
    """
    if PLATFORM_IS_WINDOWS:
        return PureWindowsPath(path)
    return PurePosixPath(path)
