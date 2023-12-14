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

from gemseo.disciplines.linear_combination import LinearCombination


@pytest.fixture()
def linear_combination() -> LinearCombination:
    """A linear combination."""
    # delta = -2 + alpha - 2*beta
    return LinearCombination(
        input_names=["alpha", "beta"],
        output_name="delta",
        input_coefficients={"alpha": 1.0, "beta": -2.0},
        offset=-2.0,
    )
