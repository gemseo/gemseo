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
"""Test scaler transformer module."""
from __future__ import annotations

import pytest
from gemseo.mlearning.transform.scaler.scaler import Scaler
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import diag
from numpy import eye
from numpy import ndarray


@pytest.fixture
def data() -> ndarray:
    """Test data."""
    return arange(30).reshape((10, 3))


def test_constructor():
    """Test constructor."""
    scaler = Scaler()
    assert scaler.name == "Scaler"
    assert allclose(scaler.offset, 0)
    assert allclose(scaler.coefficient, 1)


def test_duplicate(data):
    """Test duplicate method."""
    scaler = Scaler()
    scaler.fit(data)
    scaler_dup = scaler.duplicate()
    assert scaler != scaler_dup
    assert allclose(scaler_dup.coefficient, 1)
    assert allclose(scaler_dup.offset, 0)

    scaler = Scaler(offset=5, coefficient=3)
    scaler.fit(data)
    scaler_dup = scaler.duplicate()
    assert scaler != scaler_dup
    assert allclose(scaler_dup.coefficient, 3)
    assert allclose(scaler_dup.offset, 5)


def test_fit(data):
    """Test fit method."""
    scaler = Scaler()
    scaler.fit(data)


def test_transform(data):
    """Test transform method."""
    scaler = Scaler()
    another_scaler = Scaler(offset=3, coefficient=2)
    yet_another_scaler = Scaler(
        offset=array([5, 10, 30]), coefficient=array([1, -1, 100])
    )
    scaler.fit(data)
    another_scaler.fit(data)
    yet_another_scaler.fit(data)

    scaled = scaler.transform(data)
    another_scaled = another_scaler.transform(data)
    yet_another_scaled = yet_another_scaler.transform(data)
    assert allclose(scaled, data)
    assert allclose(another_scaled, 3 + 2 * data)
    assert allclose(yet_another_scaled, array([5, 10, 30]) + array([1, -1, 100]) * data)


def test_inverse_transform(data):
    """Test inverse_transform method."""
    scaler = Scaler()
    another_scaler = Scaler(offset=3, coefficient=2)
    yet_another_scaler = Scaler(
        offset=array([5, 10, 30]), coefficient=array([1, -1, 100])
    )
    scaler.fit(data)
    another_scaler.fit(data)
    yet_another_scaler.fit(data)

    unscaled = scaler.inverse_transform(data)
    another_unscaled = another_scaler.inverse_transform(data)
    yet_another_unscaled = yet_another_scaler.inverse_transform(data)
    assert allclose(unscaled, data)
    assert allclose(another_unscaled, (data - 3) / 2)
    assert allclose(
        yet_another_unscaled, (data - array([5, 10, 30])) / array([1, -1, 100])
    )


def test_compute_jacobian(data):
    """Test compute_jacobian method."""
    iden = eye(data.shape[1])

    scaler = Scaler()
    another_scaler = Scaler(offset=3, coefficient=2)
    yet_another_scaler = Scaler(
        offset=array([5, 10, 30]), coefficient=array([1, -1, 100])
    )
    scaler.fit(data)
    another_scaler.fit(data)
    yet_another_scaler.fit(data)

    jac = scaler.compute_jacobian(data)
    another_jac = another_scaler.compute_jacobian(data)
    yet_another_jac = yet_another_scaler.compute_jacobian(data)

    assert allclose(jac, iden)
    assert allclose(another_jac, 2 * iden)
    assert allclose(yet_another_jac, diag(array([1, -1, 100])))


def test_compute_jacobian_inverse(data):
    """Test compute_jacobian_inverse method."""
    iden = eye(data.shape[1])

    scaler = Scaler()
    another_scaler = Scaler(offset=3, coefficient=2)
    yet_another_scaler = Scaler(
        offset=array([5, 10, 30]), coefficient=array([1, -1, 100])
    )
    scaler.fit(data)
    another_scaler.fit(data)
    yet_another_scaler.fit(data)

    jac_inv = scaler.compute_jacobian_inverse(data)
    another_jac_inv = another_scaler.compute_jacobian_inverse(data)
    yet_another_jac_inv = yet_another_scaler.compute_jacobian_inverse(data)

    assert allclose(jac_inv, iden)
    assert allclose(another_jac_inv, 1 / 2 * iden)
    assert allclose(yet_another_jac_inv, diag(1 / array([1, -1, 100])))
