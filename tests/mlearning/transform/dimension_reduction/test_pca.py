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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Syver Doving Agdestein
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test principal component analysis dimension reduction."""
from __future__ import annotations

import pytest
from gemseo.mlearning.transform.dimension_reduction.pca import PCA
from numpy import allclose
from numpy import arange
from numpy import ndarray

N_SAMPLES = 10
N_FEATURES = 8


@pytest.fixture
def data() -> ndarray:
    """The dataset used to build the transformer, based on a 1D-mesh."""
    return arange(N_SAMPLES * N_FEATURES).reshape(N_SAMPLES, N_FEATURES)


def test_constructor():
    """Test constructor."""
    n_components = 3
    pca = PCA(n_components=n_components)
    assert pca.name == "PCA"
    assert pca.algo is not None
    assert pca.n_components == n_components


def test_learn(data):
    """Test learn with the default number of components (None)."""
    pca = PCA()
    pca.fit(data)
    assert pca.n_components == pca.algo.n_components_


def test_learn_custom(data):
    """Test learn with a custom number of components."""
    n_components = 3
    pca = PCA(n_components=n_components)
    pca.fit(data)
    assert pca.n_components == n_components


def test_transform(data):
    """Test transform."""
    n_components = 3
    pca = PCA(n_components=n_components)
    pca.fit(data)
    reduced_data = pca.transform(data)
    assert reduced_data.shape[0] == data.shape[0]
    assert reduced_data.shape[1] == n_components


def test_inverse_transform(data):
    """Test inverse transform."""
    n_components = 3
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data = arange(N_SAMPLES * n_components).reshape((N_SAMPLES, n_components))
    restored_data = pca.inverse_transform(data)
    assert restored_data.shape[0] == data.shape[0]
    assert restored_data.shape[1] == N_FEATURES


def test_compute_jacobian(data):
    """Test compute_jacobian method."""

    n_components = 3
    pca = PCA(n_components=n_components)
    pca.fit(data)
    jac = pca.compute_jacobian(data)

    assert jac.shape == (n_components, data.shape[1])
    assert allclose(jac, pca.algo.components_)


def test_compute_jacobian_inverse(data):
    """Test compute_jacobian_inverse method."""
    n_components = 3
    pca = PCA(n_components=n_components)
    pca.fit(data)
    jac_inv = pca.compute_jacobian_inverse(data)

    assert jac_inv.shape == (data.shape[1], n_components)
    assert allclose(jac_inv, pca.algo.components_.T)


def test_components(data):
    """Test transform."""
    n_components = 3
    pca = PCA(n_components=n_components)
    pca.fit(data)
    assert pca.components.shape[0] == data.shape[1]
    assert pca.components.shape[1] == n_components
