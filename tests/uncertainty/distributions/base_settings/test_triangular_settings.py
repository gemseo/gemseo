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

import re

import pytest

from gemseo.uncertainty.distributions.base_settings.triangular_settings import (
    BaseTriangularDistribution_Settings,
)


class SpecificTriangularDistribution_Settings(BaseTriangularDistribution_Settings):  # noqa: N801
    _TARGET_CLASS_NAME = "SpecificTriangularDistribution"


@pytest.mark.parametrize(
    ("minimum", "mode", "maximum"), [(1, 3, 2), (2, 1, 3), (3, 2, 1)]
)
def test_validator(minimum, mode, maximum):
    """Test BaseTriangularDistribution_Settings.__validate."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The parameters of the triangular distribution do not satisfy "
            "the order: minimum < mode < maximum."
        ),
    ):
        SpecificTriangularDistribution_Settings(
            minimum=minimum, mode=mode, maximum=maximum
        )
