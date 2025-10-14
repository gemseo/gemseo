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
"""NumPy utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy import dtype
    from numpy import ndarray


def convert_array_type(a: ndarray, dtype_: dtype, copy: bool = True) -> ndarray:
    """Convert an array to a specific type.

    Args:
        a: The original array.
        dtype_: The specific type.
        copy: Whether to return a copy when it is possible.

    Returns:
        The array converted to the specific type.
    """
    return (a.real if dtype_.kind == "c" else a).astype(dtype_, copy=copy)
