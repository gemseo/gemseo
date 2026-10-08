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
from typing import Any

from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

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


def format_elided(values: Sequence[Any] | ndarray, noun: str, max_length: int) -> str:
    """Return a readable representation of values, elided when there are many.

    Beyond `max_length` values,
    the representation is elided around its extremes and followed by its length,
    so that a message quoting many values stays readable.

    Args:
        values: The values.
        noun: The plural noun naming the values, e.g. `"choices"`.
        max_length: The number of values above which
            the representation is elided.
            A value below `1` is read as `1`,
            since an elided representation keeps at least one value.

    Returns:
        The representation of the values,
        e.g. `"[2.0, 4.0, 6.0, ..., 96.0, 98.0, 100.0] (50 choices)"`.
    """
    max_length = max(max_length, 1)
    if len(values) <= max_length:
        return f"[{pretty_str(values, sort=False, use_and=False)}]"

    n_head = max_length // 2
    n_tail = max_length - n_head
    tail = pretty_str(values[-n_tail:], sort=False, use_and=False)
    if not n_head:
        # The head is empty, and `values[-0:]` would be every value,
        # so render the tail alone rather than an empty leading element
        # followed by the whole list.
        return f"[..., {tail}] ({len(values)} {noun})"

    head = pretty_str(values[:n_head], sort=False, use_and=False)
    return f"[{head}, ..., {tail}] ({len(values)} {noun})"
