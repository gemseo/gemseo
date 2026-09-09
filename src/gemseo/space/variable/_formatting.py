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
"""Formatting of variable data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray


def format_components(array: ndarray, indices: Iterable[int]) -> str:
    """Return a readable representation of some components of an array.

    Args:
        array: The array.
        indices: The indices of the components,
            sorted in ascending order in the representation.

    Returns:
        The components with their indices,
        e.g. `"nan (index 0) and inf (index 2)"`.
    """
    return pretty_str(
        [f"{array[index]} (index {index})" for index in sorted(indices)], sort=False
    )
