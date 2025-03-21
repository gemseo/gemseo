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

from gemseo.problems.uncertainty.wing_weight.uncertain_space import (
    WingWeightUncertainSpace,
)


def test_wing_weight_space() -> None:
    """Check the WingWeight space."""
    space = WingWeightUncertainSpace()
    loc_values = [6, -10, 2.5, 150, 1700, 220, 0.025, 0.5, 16, 0.08]
    scale_values = [4, 20, 3.5, 50, 800, 80, 0.055, 0.5, 29, 0.1]
    assert space.dimension == 10
    assert space.variable_names == [
        "A",
        "Lamda",
        "Nz",
        "Sw",
        "Wdg",
        "Wfw",
        "Wp",
        "ell",
        "q",
        "tc",
    ]
    for distribution, loc, scale in zip(
        space.distributions.values(), loc_values, scale_values
    ):
        assert len(distribution.marginals) == 1
        distribution = distribution.marginals[0].distribution
        assert distribution.dist.__class__.__name__ == "uniform_gen"
        assert distribution.kwds == pytest.approx({
            "loc": loc,
            "scale": scale,
        })
