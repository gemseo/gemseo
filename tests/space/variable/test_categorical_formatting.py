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
"""Tests for the formatting of the categories of a categorical variable."""

from __future__ import annotations

import pytest

from gemseo.space.variable import CategoricalVariable


@pytest.mark.parametrize(
    ("categories", "expected"),
    [
        (["a", "b"], "[a, b]"),
        (list("abcdef"), "[a, b, c, d, e, f]"),
        (list("abcdefg"), "[a, b, c, ..., e, f, g] (7 categories)"),
    ],
)
def test_format_categories(categories, expected) -> None:
    """Check the rendering of the categories, elided beyond six labels."""
    variable = CategoricalVariable(categories=categories)
    assert variable._format_categories() == expected
