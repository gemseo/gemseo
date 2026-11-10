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

"""The squarified layout of a treemap."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final
from typing import NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

_relative_area_tolerance: Final[float] = 1e-12
"""The share of the total area below which an area is considered as zero."""


class Rectangle(NamedTuple):
    """An axis-aligned rectangle."""

    x: float
    """The abscissa of the corner with the smallest coordinates."""

    y: float
    """The ordinate of the corner with the smallest coordinates."""

    width: float
    """The extent along the abscissa."""

    height: float
    """The extent along the ordinate."""


def compute_squarified_layout(
    areas: Sequence[float], width: float, height: float
) -> list[Rectangle]:
    """Compute a squarified treemap layout of areas in a rectangle.

    The areas are rescaled so that their sum is `width * height`.
    The tiles are arranged in rows along the shorter side of the remaining space.
    A row keeps growing while the worst aspect ratio of its tiles does not get worse.

    Args:
        areas: The areas of the tiles, in the order of placement.
        width: The width of the rectangle to be filled.
        height: The height of the rectangle to be filled.

    Returns:
        The rectangles, in the order of `areas`.
        A tile whose area is zero,
        or a negligible share of the total area,
        gets a rectangle of zero size.
    """
    rectangles = [Rectangle(0.0, 0.0, 0.0, 0.0)] * len(areas)
    total_area = sum(areas)
    scale = width * height / total_area
    min_area = total_area * _relative_area_tolerance
    indices = [index for index in range(len(areas)) if areas[index] > min_area]
    scaled_areas = [areas[index] * scale for index in indices]
    x, y = 0.0, 0.0
    free_width, free_height = float(width), float(height)
    start = 0
    while start < len(indices):
        # The row is a column along the left side if the free space is wider
        # than high, otherwise a line along the top side.
        vertical = free_width >= free_height
        side = free_height if vertical else free_width
        if side <= 0:
            # The rounding errors have consumed the free space;
            # the remaining tiles keep their rectangles of zero size.
            break

        n_tiles, row_area = _count_tiles_of_row(scaled_areas, start, side)
        stop = start + n_tiles
        if stop == len(indices):
            # The last row takes all the free space, whatever the rounding errors.
            thickness = free_width if vertical else free_height
        else:
            thickness = row_area / side

        row_start = y if vertical else x
        position = row_start
        for k in range(start, stop):
            if k == stop - 1:
                # The last tile ends exactly on the border of the free space.
                length = row_start + side - position
            else:
                length = scaled_areas[k] / thickness
            if vertical:
                rectangles[indices[k]] = Rectangle(x, position, thickness, length)
            else:
                rectangles[indices[k]] = Rectangle(position, y, length, thickness)

            position += length

        if vertical:
            x += thickness
            free_width -= thickness
        else:
            y += thickness
            free_height -= thickness

        start = stop

    return rectangles


def _count_tiles_of_row(
    areas: Sequence[float], start: int, side: float
) -> tuple[int, float]:
    """Count the tiles to be put in the next row.

    Args:
        areas: The areas of the tiles, in the order of placement.
        start: The index of the first tile of the row.
        side: The length of the side along which the row is laid out.

    Returns:
        The number of tiles of the row,
        which is the largest one for which the worst aspect ratio
        does not get worse when a tile is added,
        and the sum of their areas.
    """
    # The rule of the squarify algorithm of Bruls, Huizing & van Wijk (2000),
    # Section 3: a tile is added to the current row
    # as long as it does not make the worst aspect ratio of the row worse;
    # otherwise the row is closed and the tile starts a new one.
    squared_side = side * side
    largest_area = smallest_area = row_area = areas[start]
    worst_ratio = _compute_worst_aspect_ratio(
        row_area, largest_area, smallest_area, squared_side
    )
    n_tiles = 1
    for index in range(start + 1, len(areas)):
        area = areas[index]
        new_row_area = row_area + area
        new_largest_area = max(largest_area, area)
        new_smallest_area = min(smallest_area, area)
        new_ratio = _compute_worst_aspect_ratio(
            new_row_area, new_largest_area, new_smallest_area, squared_side
        )
        if new_ratio > worst_ratio:
            break

        row_area = new_row_area
        largest_area = new_largest_area
        smallest_area = new_smallest_area
        worst_ratio = new_ratio
        n_tiles += 1

    return n_tiles, row_area


def _compute_worst_aspect_ratio(
    row_area: float, largest_area: float, smallest_area: float, squared_side: float
) -> float:
    """Compute the worst aspect ratio of the tiles of a row.

    Args:
        row_area: The sum of the areas of the tiles of the row.
        largest_area: The largest area of the row.
        smallest_area: The smallest area of the row.
        squared_side: The square of the length of the side
            along which the row is laid out.

    Returns:
        The worst aspect ratio, which is greater than or equal to 1.
    """
    # The function worst(R, w) of Bruls, Huizing & van Wijk (2000), Section 3,
    # i.e. max(w^2 r+ / s^2, s^2 / (w^2 r-))
    # for a row R of total area s, largest area r+ and smallest area r-
    # laid out along a side of length w.
    return max(
        squared_side * largest_area / row_area**2,
        row_area**2 / (squared_side * smallest_area),
    )
