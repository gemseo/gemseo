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
"""Test min-max scaler module."""

from __future__ import annotations

import pytest
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import ndarray
from numpy import zeros
from numpy.testing import assert_almost_equal

from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler


@pytest.fixture()
def data() -> ndarray:
    """Test data."""
    return arange(30).reshape((10, 3))


def test_constructor():
    """Test constructor."""
    scaler = MinMaxScaler()
    assert scaler.name == "MinMaxScaler"


def test_fit(data):
    """Test fit method."""
    left = data.min(0)
    right = data.max(0)
    scaler = MinMaxScaler()
    scaler.fit(data)
    assert allclose(scaler.offset, -left / (right - left))
    assert allclose(scaler.coefficient, 1 / (data.max(0) - data.min(0)))


def test_transform(data):
    """Test transform method."""
    left = data.min(0)
    right = data.max(0)
    scaler = MinMaxScaler()
    another_scaler = MinMaxScaler(offset=3, coefficient=2)
    scaler.fit(data)
    another_scaler.fit(data)  # fit() should overload parameters
    scaled_data = scaler.transform(data)
    other_scaled_data = another_scaler.transform(data)
    assert allclose(scaled_data, (data - left) / (right - left))
    assert allclose(other_scaled_data, (data - left) / (right - left))


def test_inverse_transform(data):
    """Test inverse_transform method."""
    left = data.min(0)
    right = data.max(0)
    scaler = MinMaxScaler()
    another_scaler = MinMaxScaler(offset=3, coefficient=2)
    scaler.fit(data)
    another_scaler.fit(data)  # fit() should overload parameters
    unscaled_data = scaler.inverse_transform(data)
    other_unscaled_data = another_scaler.inverse_transform(data)
    assert allclose(unscaled_data, left + (right - left) * data)
    assert allclose(other_unscaled_data, left + (right - left) * data)


def _test_transformer(data, transformed_data, coefficient, offset):
    """Test the MinMaxScaler transformer.

    Args:
        data: The data to be transformed.
        transformed_data: The expected transformed data.
        coefficient: The expected coefficient.
        offset: The expected offset.
    """
    transformer = MinMaxScaler()
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
        offset = array([0.5])
        coefficient = array([1])
    else:
        offset = array([-0.5])
        coefficient = 1 / data[0]
    transformed_data = offset + coefficient * data
    _test_transformer(data, transformed_data, coefficient, offset)


def test_with_constant():
    """Check scaling with constant feature."""
    offset = zeros(4)
    coefficient = zeros(4)
    data = array([[1.0, 2.0, 0.0, 6.0], [4.0, 2.0, 0.0, 3.0]])
    minimum = data[:, [0, 3]].min(0)
    delta = data[:, [0, 3]].max(0) - minimum
    offset[[0, 3]] = -minimum / delta
    coefficient[[0, 3]] = 1 / delta
    offset[[1, 2]] = array([-0.5, 0.5])
    coefficient[[1, 2]] = array([1 / data[0, 1], 1])
    transformed_data = offset + coefficient * data
    _test_transformer(data, transformed_data, coefficient, offset)


@pytest.mark.parametrize(
    "data",
    [array([[1.0, 2.0, 6.0], [4.0, 5.0, 3.0]]), array([[2.0], [4.0]])],
)
def test_without_constant(data):
    """Check scaling without constant feature."""
    minimum = data.min(0)
    delta = data.max(0) - minimum
    offset = -minimum / delta
    coefficient = 1 / delta
    transformed_data = offset + coefficient * data
    _test_transformer(data, transformed_data, coefficient, offset)
