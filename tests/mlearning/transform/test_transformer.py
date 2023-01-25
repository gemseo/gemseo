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
"""Test transformer module."""
from __future__ import annotations

from unittest import mock

import pytest
from gemseo.mlearning.transform.transformer import Transformer
from gemseo.utils.pytest_conftest import concretize_classes
from gemseo.utils.python_compatibility import get_mock_method_call_args
from numpy import arange
from numpy import array
from numpy import ndarray


@pytest.fixture
def data() -> ndarray:
    """Test data."""
    return arange(30).reshape((10, 3))


def test_constructor():
    """Test constructor."""
    with concretize_classes(Transformer):
        transformer = Transformer()
        another_transformer = Transformer("Another Transformer")

    assert transformer.name == "Transformer"
    assert another_transformer.name == "Another Transformer"
    assert not transformer.is_fitted


def test_fit(data):
    """Test fit method.

    fit calls _fit which is an abstract method and then sets is_fitted as True.
    """
    with concretize_classes(Transformer):
        transformer = Transformer()

    transformer._fit = lambda data, *args: None
    transformer.fit(array([1]))
    assert transformer.is_fitted


def test_str():
    """Test string representation."""
    with concretize_classes(Transformer):
        transformer = Transformer()

    assert str(transformer) == "Transformer"


@pytest.mark.parametrize("data", [array([1, 1, 1]), array([[1], [1], [1]])])
def test_fit_shape(data):
    """Check that fit() handles both 1D and 2D NumPy arrays."""
    with concretize_classes(Transformer):
        transformer = Transformer()
        with mock.patch.object(transformer, "_fit") as mock_method:
            transformer.fit(data)

        assert get_mock_method_call_args(mock_method)[0].shape == (3, 1)
