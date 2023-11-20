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
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import diag
from numpy import ndarray
from numpy import tile

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.mlearning.transformers.dimension_reduction.pca import PCA

N_SAMPLES = 10
N_FEATURES = 8
N_COMPONENTS = 3


@pytest.fixture(scope="module")
def data() -> ndarray:
    """The dataset used to build the transformer, based on a 1D-mesh."""
    return arange(N_SAMPLES * N_FEATURES).reshape(N_SAMPLES, N_FEATURES)


@pytest.fixture(scope="module", params=[False, True])
def pca(request, data) -> PCA:
    """A PCA with or without data scaling."""
    if request.param:
        pca = PCA(n_components=N_COMPONENTS, scale=True)
    else:
        pca = PCA(n_components=N_COMPONENTS)
    pca.fit(data)
    return pca


def test_constructor():
    """Test constructor."""
    pca = PCA(n_components=N_COMPONENTS)
    assert pca.name == "PCA"
    assert pca.algo is not None
    assert pca.n_components == N_COMPONENTS
    assert not pca.data_is_scaled
    assert pca._PCA__scaler.__class__.__name__ == "Scaler"


def test_scales_data():
    """Test data_is_scaled."""
    pca = PCA(n_components=N_COMPONENTS, scale=True)
    assert pca.data_is_scaled
    assert pca._PCA__scaler.__class__.__name__ == "StandardScaler"


def test_learn(pca):
    """Test learn with the default number of components (None)."""
    assert pca.n_components == pca.algo.n_components_


def test_learn_custom(pca):
    """Test learn with a custom number of components."""
    assert pca.n_components == N_COMPONENTS


def test_transform(data, pca):
    """Test transform."""
    reduced_data = pca.transform(data)
    assert reduced_data.shape[0] == data.shape[0]
    assert reduced_data.shape[1] == N_COMPONENTS


def test_inverse_transform(pca):
    """Test inverse transform."""
    data = arange(N_SAMPLES * N_COMPONENTS).reshape((N_SAMPLES, N_COMPONENTS))
    restored_data = pca.inverse_transform(data)
    assert restored_data.shape[0] == data.shape[0]
    assert restored_data.shape[1] == N_FEATURES


def test_compute_jacobian(data, pca):
    """Test compute_jacobian method."""
    jac = pca.compute_jacobian(data)
    assert jac.shape == (data.shape[0], N_COMPONENTS, data.shape[1])
    shape = (len(data), 1, 1)
    expectation = tile(pca.algo.components_, shape)
    if pca.data_is_scaled:
        coefficient = 1 / data.std(0)
        expectation = expectation @ tile(diag(coefficient), shape)
    assert allclose(jac, expectation)


def test_compute_jacobian_inverse(data, pca):
    """Test compute_jacobian_inverse method."""
    transformed_data = pca.transform(data)
    jac_inv = pca.compute_jacobian_inverse(transformed_data)
    assert jac_inv.shape == (data.shape[0], data.shape[1], N_COMPONENTS)
    shape = (len(data), 1, 1)
    expectation = tile(pca.algo.components_.T, shape)
    if pca.data_is_scaled:
        coefficient = 1 / data.std(0)
        expectation = tile(diag(1 / coefficient), shape) @ expectation

    assert allclose(jac_inv, expectation)


def test_components(pca):
    """Test transform."""
    assert pca.components.shape[0] == N_FEATURES
    assert pca.components.shape[1] == N_COMPONENTS


def test_shape(data, pca):
    """Check the shapes of the data."""
    n = N_SAMPLES
    p = N_FEATURES
    q = N_COMPONENTS
    transformed_data = pca.transform(data)
    assert transformed_data.shape == (n, q)
    assert pca.inverse_transform(transformed_data).shape == (n, p)

    assert pca.transform(data[0]).shape == (q,)
    assert pca.inverse_transform(transformed_data[0]).shape == (p,)

    assert pca.compute_jacobian(data).shape == (n, q, p)
    assert pca.compute_jacobian_inverse(transformed_data).shape == (n, p, q)

    assert pca.compute_jacobian(data[0]).shape == (q, p)
    assert pca.compute_jacobian_inverse(transformed_data[0]).shape == (p, q)


def test_transformation_jacobian(pca):
    """Check the Jacobian of the transformation."""
    function = MDOFunction(pca.transform, "transform", jac=pca.compute_jacobian)
    input_data = array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    function.check_grad(input_data, error_max=1e-7)
    function.check_grad(input_data[::-1], error_max=1e-7)


def test_inverse_transformation_jacobian(pca):
    """Check the Jacobian of the inverse transformation."""
    function = MDOFunction(
        pca.inverse_transform, "inverse_transform", jac=pca.compute_jacobian_inverse
    )
    input_data = array([1.0, 2.0, 3.0])
    function.check_grad(input_data, error_max=1e-7)
    function.check_grad(input_data[::-1], error_max=1e-7)
