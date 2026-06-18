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
from __future__ import annotations

import pytest

from gemseo.utils.string_tools import convert_camel_case_to_screaming_snake_case


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # The names of the dynamically built enums whose conversion this guards.
        ("ProgressBarData", "PROGRESS_BAR_DATA"),
        ("Dataset", "DATASET"),
        ("IODataset", "IO_DATASET"),
        ("OptimizationDataset", "OPTIMIZATION_DATASET"),
        ("Arcsine", "ARCSINE"),
        ("ChiSquare", "CHI_SQUARE"),
        ("FisherSnedecor", "FISHER_SNEDECOR"),
        ("GeneralizedPareto", "GENERALIZED_PARETO"),
        ("InverseNormal", "INVERSE_NORMAL"),
        # Acronym kept together, then split from the following word.
        ("HTMLParser", "HTML_PARSER"),
        # Trailing digits stay attached to the preceding word.
        ("Sobieski2", "SOBIESKI2"),
        # Already screaming snake case is left unchanged.
        ("ALREADY_SNAKE", "ALREADY_SNAKE"),
    ],
)
def test_convert_camel_case_to_screaming_snake_case(name: str, expected: str) -> None:
    """Verify the conversion of a camel case string to screaming snake case."""
    assert convert_camel_case_to_screaming_snake_case(name) == expected
