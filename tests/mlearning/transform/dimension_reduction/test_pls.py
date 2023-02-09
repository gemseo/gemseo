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
"""Test partial least square regression."""
from __future__ import annotations

import pytest
from gemseo.mlearning.transform.dimension_reduction.pls import PLS
from numpy import ndarray
from numpy import newaxis
from numpy import sum as npsum
from numpy.random import rand

N_SAMPLES = 10
N_FEATURES = 8


@pytest.fixture
def data() -> tuple[ndarray, ndarray]:
    """The dataset used to build the transformer, based on a 1D-mesh."""
    input_data = rand(N_SAMPLES, N_FEATURES)
    output_data = npsum(input_data, 1)[:, newaxis]
    return input_data, output_data


def test_constructor():
    """Test constructor."""
    n_components = 3
    pca = PLS(n_components=n_components)
    assert pca.name == "PLS"
    assert pca.algo is not None
    assert pca.n_components == n_components


def test_learn(data):
    """Test learn with the default number of components (None)."""
    input_data, output_data = data
    pca = PLS()
    pca.fit(input_data, output_data)
    assert pca.n_components == 1


def test_learn_custom(data):
    """Test learn with a custom number of components."""
    input_data, output_data = data
    n_components = 3
    pls = PLS(n_components=n_components)
    pls.fit(input_data, output_data)
    assert pls.n_components == n_components


def test_transform(data):
    """Test transform."""
    input_data, output_data = data
    n_components = 3
    pca = PLS(n_components=n_components)
    pca.fit(input_data, output_data)
    reduced_data = pca.transform(input_data)
    assert reduced_data.shape[0] == input_data.shape[0]
    assert reduced_data.shape[1] == n_components


def test_inverse_transform(data):
    """Test inverse transform."""
    input_data, output_data = data
    n_components = 3
    pca = PLS(n_components=n_components)
    pca.fit(input_data, output_data)
    data = rand(N_SAMPLES, n_components)
    restored_data = pca.inverse_transform(data)
    assert restored_data.shape[0] == data.shape[0]
    assert restored_data.shape[1] == N_FEATURES


def test_shape(data):
    """Check the shapes of the data."""
    input_data, output_data = data
    pls = PLS(n_components=3)
    pls.fit(input_data, output_data)
    n, p = input_data.shape
    q = pls.n_components
    transformed_data = pls.transform(input_data)
    assert transformed_data.shape == (n, q)
    assert pls.inverse_transform(transformed_data).shape == (n, p)

    assert pls.transform(input_data[0]).shape == (q,)
    assert pls.inverse_transform(transformed_data[0]).shape == (p,)
