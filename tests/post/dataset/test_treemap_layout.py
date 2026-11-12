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
"""Test the squarified layout of a treemap."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy.random import default_rng

from gemseo.post.dataset.plot import _treemap_layout
from gemseo.post.dataset.plot._treemap_layout import Rectangle
from gemseo.post.dataset.plot._treemap_layout import compute_squarified_layout

if TYPE_CHECKING:
    from collections.abc import Sequence

AREAS = [6, 6, 4, 3, 2, 2, 1]


def _assert_partition(
    rectangles: Sequence[Rectangle], width: float, height: float
) -> None:
    """Check that the rectangles lie in the container and cover its area."""
    for rectangle in rectangles:
        assert rectangle.x >= -1e-12
        assert rectangle.y >= -1e-12
        assert rectangle.x + rectangle.width <= width + 1e-12
        assert rectangle.y + rectangle.height <= height + 1e-12

    assert sum(r.width * r.height for r in rectangles) == pytest.approx(width * height)


def test_compute_squarified_layout_paper_example() -> None:
    """Check the layout of areas 6, 6, 4, 3, 2, 2, 1 in a 6x4 rectangle.

    The expected rectangles come from applying the algorithm by hand:
    the 1st row is a column of width 3 on the left side (6 and 6),
    the 2nd one a line of height 7/3 (4 and 3),
    then come two columns of widths 6/5 (2) and 6/5 (2)
    and a last one of width 3/5 (1) in the remaining space.
    """
    expected = [
        Rectangle(0.0, 0.0, 3.0, 2.0),
        Rectangle(0.0, 2.0, 3.0, 2.0),
        Rectangle(3.0, 0.0, 12 / 7, 7 / 3),
        Rectangle(3.0 + 12 / 7, 0.0, 9 / 7, 7 / 3),
        Rectangle(3.0, 7 / 3, 6 / 5, 5 / 3),
        Rectangle(4.2, 7 / 3, 6 / 5, 5 / 3),
        Rectangle(5.4, 7 / 3, 3 / 5, 5 / 3),
    ]
    rectangles = compute_squarified_layout(AREAS, 6, 4)
    assert rectangles == [pytest.approx(rectangle) for rectangle in expected]


@pytest.mark.parametrize(
    ("width", "height"),
    [
        pytest.param(6.0, 4.0, id="wide"),
        pytest.param(1.0, 1.0, id="unit_square"),
        pytest.param(2.0, 10.0, id="tall"),
    ],
)
def test_compute_squarified_layout_properties(width, height) -> None:
    """Check the areas, the containment in the rectangle and the absence of overlap."""
    rectangles = compute_squarified_layout(AREAS, width, height)
    total_area = sum(AREAS)
    for area, rectangle in zip(AREAS, rectangles, strict=True):
        # The result follows the order of the input.
        assert rectangle.width * rectangle.height == pytest.approx(
            area * width * height / total_area
        )

    _assert_partition(rectangles, width, height)

    # The last tile ends on the border of the rectangle.
    last = rectangles[-1]
    assert last.x + last.width == pytest.approx(width)
    assert last.y + last.height == pytest.approx(height)

    for i, first in enumerate(rectangles):
        for second in rectangles[i + 1 :]:
            overlap_width = min(first.x + first.width, second.x + second.width) - max(
                first.x, second.x
            )
            overlap_height = min(
                first.y + first.height, second.y + second.height
            ) - max(first.y, second.y)
            assert overlap_width <= 1e-12 or overlap_height <= 1e-12


def test_compute_squarified_layout_input_order() -> None:
    """Check that the rectangles follow the order of the areas.

    The first area given is laid out at the origin.
    """
    rectangles = compute_squarified_layout([1, 6, 2], 3, 2)
    assert [r.width * r.height for r in rectangles] == pytest.approx([2 / 3, 4, 4 / 3])
    assert rectangles[0].x == rectangles[0].y == 0.0


def test_compute_squarified_layout_one_tile() -> None:
    """Check that a single tile fills the rectangle."""
    assert compute_squarified_layout([3.0], 5, 2) == [Rectangle(0.0, 0.0, 5.0, 2.0)]


def test_compute_squarified_layout_zero_areas() -> None:
    """Check that zero areas give rectangles of zero size at their input position."""
    rectangles = compute_squarified_layout([0.0, 1.0, 0.0, 1.0], 2, 1)
    assert rectangles[0] == Rectangle(0.0, 0.0, 0.0, 0.0)
    assert rectangles[2] == Rectangle(0.0, 0.0, 0.0, 0.0)
    assert rectangles[1].width * rectangles[1].height == pytest.approx(1.0)
    assert rectangles[3].width * rectangles[3].height == pytest.approx(1.0)


def test_compute_squarified_layout_last_area_last() -> None:
    """Check that a tile given last is placed last, even if it is not the smallest."""
    rectangles = compute_squarified_layout([0.5, 0.1, 0.4], 1, 1)
    assert rectangles == [
        pytest.approx(Rectangle(0.0, 0.0, 0.5, 1.0)),
        pytest.approx(Rectangle(0.5, 0.0, 0.5, 0.2)),
        pytest.approx(Rectangle(0.5, 0.2, 0.5, 0.8)),
    ]


@pytest.mark.parametrize(
    ("areas", "tolerance"),
    [
        pytest.param([1, 1.1e-16], None, id="default_tolerance-one_and_epsilon"),
        pytest.param([1, 1e-17], None, id="default_tolerance-one_and_tiny"),
        pytest.param(
            [0.5, 0.5, 1e-17], None, id="default_tolerance-two_halves_and_tiny"
        ),
        pytest.param(
            [0.6, 0.3, 0.1, 1e-17], None, id="default_tolerance-three_areas_and_tiny"
        ),
        pytest.param([1, 1.1e-16], 0.0, id="no_tolerance-one_and_epsilon"),
        pytest.param([1, 1e-17], 0.0, id="no_tolerance-one_and_tiny"),
        pytest.param([0.5, 0.5, 1e-17], 0.0, id="no_tolerance-two_halves_and_tiny"),
    ],
)
def test_compute_squarified_layout_negligible_area(
    areas, tolerance, monkeypatch
) -> None:
    """Check that a negligible area gets a rectangle of zero size.

    Without tolerance,
    the tiles placed before the negligible one leave no free space,
    which holds only when their rows fill the rectangle exactly.
    """
    if tolerance is not None:
        monkeypatch.setattr(_treemap_layout, "_relative_area_tolerance", tolerance)

    rectangles = compute_squarified_layout(areas, 1, 1)
    assert rectangles[-1] == Rectangle(0.0, 0.0, 0.0, 0.0)
    assert all(r.width >= 0 and r.height >= 0 for r in rectangles)
    assert sum(r.width * r.height for r in rectangles[:-1]) == pytest.approx(1.0)
    for area, rectangle in zip(areas[:-1], rectangles[:-1], strict=True):
        assert rectangle.width * rectangle.height == pytest.approx(area / sum(areas))


def test_compute_squarified_layout_stays_in_unit_square() -> None:
    """Check that the rounding errors never push a tile out of the unit square."""
    rng = default_rng(0)
    for _ in range(200):
        areas = (10 ** rng.uniform(-4.0, 0.0, rng.integers(1, 31))).tolist()
        _assert_partition(compute_squarified_layout(areas, 1, 1), 1, 1)
