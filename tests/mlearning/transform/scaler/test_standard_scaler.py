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
"""Test standard scaler module."""
from __future__ import annotations

import pytest
from gemseo.mlearning.transform.scaler.standard_scaler import StandardScaler
from numpy import allclose
from numpy import arange
from numpy import mean as npmean
from numpy import ndarray
from numpy import std as npstd


@pytest.fixture
def data() -> ndarray:
    """Test data."""
    return arange(30).reshape((10, 3))


def test_constructor():
    """Test constructor."""
    scaler = StandardScaler()
    assert scaler.name == "StandardScaler"


def test_fit(data):
    """Test fit method."""
    mean = npmean(data, 0)
    std = npstd(data, 0)
    scaler = StandardScaler()
    scaler.fit(data)
    assert allclose(scaler.offset, -mean / std)
    assert allclose(scaler.coefficient, 1 / std)


def test_transform(data):
    """Test transform method."""
    mean = npmean(data, 0)
    std = npstd(data, 0)
    scaler = StandardScaler()
    another_scaler = StandardScaler(offset=3, coefficient=2)
    scaler.fit(data)
    another_scaler.fit(data)  # fit() should overload parameters
    scaled_data = scaler.transform(data)
    other_scaled_data = another_scaler.transform(data)
    assert allclose(scaled_data, (data - mean) / std)
    assert allclose(other_scaled_data, (data - mean) / std)


def test_inverse_transform(data):
    """Test inverse_transform method."""
    mean = npmean(data, 0)
    std = npstd(data, 0)
    scaler = StandardScaler()
    another_scaler = StandardScaler(offset=3, coefficient=2)
    scaler.fit(data)
    another_scaler.fit(data)  # fit() should overload parameters
    unscaled_data = scaler.inverse_transform(data)
    other_unscaled_data = another_scaler.inverse_transform(data)
    assert allclose(unscaled_data, mean + std * data)
    assert allclose(other_unscaled_data, mean + std * data)
