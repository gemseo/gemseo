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
from __future__ import annotations

import pytest
from gemseo.mlearning.transform.sensor.jameson import JamesonSensor
from numpy import allclose
from numpy import arange
from numpy import ndarray


@pytest.fixture
def data() -> ndarray:
    """Test data."""
    return arange(300).reshape((3, 100))


def test_constructor():
    """Test constructor."""
    sensor = JamesonSensor()
    assert sensor.name == "JamesonSensor"
    assert sensor.threshold == 0.3
    assert sensor.removing_part == 0.01
    assert sensor.dimension == 1


def test_fit(data):
    """Test fit method."""
    sensor = JamesonSensor()
    sensor.fit(data)
    assert allclose(sensor.threshold, 89.7)


def test_transform(data):
    """Test transform method."""
    sensor = JamesonSensor()
    sensor.fit(data)
    sensored = sensor.transform(data)
    assert sensored.shape == (3, 97)
    sensored = sensor.transform(data[0])
    assert sensored.shape == (97,)
