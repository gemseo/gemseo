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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test dimension reduction transformer module."""
from __future__ import annotations

import pytest
from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from numpy import arange


def test_constructor():
    """Test constructor."""
    dimred = DimensionReduction()
    assert dimred.name == "DimensionReduction"


def test_not_implemented():
    """Test not implemented methods."""
    data = arange(50).reshape((10, 5))
    dimred = DimensionReduction()
    with pytest.raises(NotImplementedError):
        dimred.fit(data)
    with pytest.raises(NotImplementedError):
        dimred.transform(data)
    with pytest.raises(NotImplementedError):
        dimred.inverse_transform(data)
