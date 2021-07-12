# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

import pytest
from numpy import arange, ndarray

from gemseo.mlearning.transform.transformer import Transformer


@pytest.fixture
def data():  # type: (...) -> ndarray
    """Test data."""
    return arange(30).reshape((10, 3))


def test_constructor():
    """Test constructor."""
    transformer = Transformer()
    another_transformer = Transformer("Another Transformer")
    assert transformer.name is not None
    assert another_transformer.name == "Another Transformer"


def test_fit(data):
    """Test fit method."""
    transformer = Transformer()
    with pytest.raises(NotImplementedError):
        transformer.fit(data)


def test_transform(data):
    """Test transform method."""
    transformer = Transformer()
    with pytest.raises(NotImplementedError):
        transformer.transform(data)


def test_inverse_transform(data):
    """Test inverse_transform method."""
    transformer = Transformer()
    with pytest.raises(NotImplementedError):
        transformer.inverse_transform(data)


def test_fit_transform(data):
    """Test fit_transform method."""
    transformer = Transformer()
    with pytest.raises(NotImplementedError):
        transformer.fit_transform(data)


def test_compute_jacobian(data):
    """Test inverse_transform method."""
    transformer = Transformer()
    with pytest.raises(NotImplementedError):
        transformer.compute_jacobian(data)


def test_compute_jacobian_inverse(data):
    """Test fit_transform method."""
    transformer = Transformer()
    with pytest.raises(NotImplementedError):
        transformer.compute_jacobian_inverse(data)


def test_str():
    """Test string representation."""
    transformer = Transformer()
    repres = str(transformer)
    assert "Transformer" in repres
