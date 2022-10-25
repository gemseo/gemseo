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
#        :author: Gilberto Ruiz Jimenez
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test the Yeo-Johnson transformer."""
from __future__ import annotations

import pytest
from gemseo.mlearning.transform.power.yeo_johnson import YeoJohnson
from numpy import allclose
from numpy import arange
from numpy import ndarray
from sklearn.preprocessing import PowerTransformer


@pytest.fixture
def data() -> ndarray:
    """Test data."""
    return arange(1.0, 31.0, 1.0).reshape((10, 3))


def test_constructor():
    """Test the constructor."""
    transform = YeoJohnson()
    assert transform.name == "YeoJohnson"


def test_fit(data):
    """Test the fit method.

    Args:
        data: The test data.
    """
    transformer = YeoJohnson()
    transformer.fit(data)

    sk_transformer = PowerTransformer(method=transformer._TRANSFORMER_NAME).fit(data)

    assert allclose(transformer.lambdas_, sk_transformer.lambdas_)


def test_transform(data):
    """Test the transform method.

    Args:
        data: The test data.
    """
    transformer = YeoJohnson()
    transformer.fit(data)
    transformed_data = transformer.transform(data)

    sk_transformed_data = PowerTransformer(
        method=transformer._TRANSFORMER_NAME
    ).fit_transform(data)

    assert allclose(transformed_data, sk_transformed_data)


def test_inverse_transform(data):
    """Test the inverse_transform method.

    Args:
        data: The test data.
    """
    transformer = YeoJohnson()
    transformer.fit(data)

    transformed_data = transformer.transform(data)

    original_data = transformer.inverse_transform(transformed_data)

    assert allclose(original_data, data)
