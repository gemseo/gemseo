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
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import ndarray
from numpy import zeros
from numpy.testing import assert_almost_equal

from gemseo.mlearning.transformers.scaler.standard_scaler import StandardScaler


@pytest.fixture()
def data() -> ndarray:
    """Test data."""
    return arange(30).reshape((10, 3))


def test_constructor():
    """Test constructor."""
    scaler = StandardScaler()
    assert scaler.name == "StandardScaler"


def test_fit(data):
    """Test fit method."""
    _mean = data.mean(0)
    _std = data.std(0)
    scaler = StandardScaler()
    scaler.fit(data)
    assert allclose(scaler.offset, -_mean / _std)
    assert allclose(scaler.coefficient, 1 / _std)


def test_transform(data):
    """Test transform method."""
    _mean = data.mean(0)
    _std = data.std(0)
    scaler = StandardScaler()
    another_scaler = StandardScaler(offset=3, coefficient=2)
    scaler.fit(data)
    another_scaler.fit(data)  # fit() should overload parameters
    scaled_data = scaler.transform(data)
    other_scaled_data = another_scaler.transform(data)
    assert allclose(scaled_data, (data - _mean) / _std)
    assert allclose(other_scaled_data, (data - _mean) / _std)


def test_inverse_transform(data):
    """Test inverse_transform method."""
    _mean = data.mean(0)
    _std = data.std(0)
    scaler = StandardScaler()
    another_scaler = StandardScaler(offset=3, coefficient=2)
    scaler.fit(data)
    another_scaler.fit(data)  # fit() should overload parameters
    unscaled_data = scaler.inverse_transform(data)
    other_unscaled_data = another_scaler.inverse_transform(data)
    assert allclose(unscaled_data, _mean + _std * data)
    assert allclose(other_unscaled_data, _mean + _std * data)


def _test_transformer(data, transformed_data, coefficient, offset):
    """Test the StandardScaler transformer.

    Args:
        data: The data to be transformed.
        transformed_data: The expected transformed data.
        coefficient: The expected coefficient.
        offset: The expected offset.
    """
    transformer = StandardScaler()
    assert_almost_equal(transformer.fit_transform(data), transformed_data)
    assert_almost_equal(transformer.inverse_transform(transformed_data), data)
    assert_almost_equal(transformer.coefficient, coefficient)
    assert_almost_equal(transformer.offset, offset)


@pytest.mark.parametrize(
    "data",
    [array([[2.0], [2.0]]), array([[0.0], [0.0]])],
)
def test_with_only_constant(data):
    """Check scaling with only constant features."""
    if data[0] == 0:
        offset = array([0])
        coefficient = array([1])
    else:
        offset = array([-1])
        coefficient = 1 / data[0]
    transformed_data = offset + coefficient * data
    _test_transformer(data, transformed_data, coefficient, offset)


def test_with_constant(data):
    """Check scaling with constant features."""
    offset = zeros(4)
    coefficient = zeros(4)
    data = array([[1.0, 2.0, 0.0, 6.0], [2.0, 2.0, 0.0, 2.0]])
    std = data[:, [0, 3]].std(0)
    offset[[0, 3]] = -data[:, [0, 3]].mean(0) / std
    coefficient[[0, 3]] = 1 / std
    offset[[1, 2]] = array([-1, 0])
    coefficient[[1, 2]] = array([1 / data[0, 1], 1])
    transformed_data = offset + coefficient * data
    _test_transformer(data, transformed_data, coefficient, offset)


@pytest.mark.parametrize(
    "data",
    [array([[1.0, 2.0, 6.0], [2.0, 5.0, 2.0]]), array([[2.0], [4.0]])],
)
def test_without_constant(data):
    """Check scaling without constant feature."""
    std = data.std(0)
    offset = -data.mean(0) / std
    coefficient = 1 / std
    transformed_data = offset + coefficient * data
    _test_transformer(data, transformed_data, coefficient, offset)
