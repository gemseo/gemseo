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
"""Test dimension reduction with Karhunen-Loeve singular value decomposition."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import linspace
from numpy import pi
from numpy import sin
from numpy.random import default_rng
from openturns import ResourceMap

from gemseo.mlearning.transformers.dimension_reduction.klsvd import KLSVD

N_SAMPLES = 100

MESH_SIZE = 10
MESH = [[t_i] for t_i in linspace(0, 1, MESH_SIZE)]
MESH2D = [
    [x_i, x_j] for x_i in linspace(0, 1, MESH_SIZE) for x_j in linspace(0, 1, MESH_SIZE)
]

RNG = default_rng()


def func(tau, theta):
    """Data generating function."""
    return sin(2 * pi * (tau - theta)) + 1


@pytest.fixture
def data():
    """The dataset used to build the transformer, based on a 1D-mesh."""
    return array([
        func(array(MESH).flatten(), theta) for theta in RNG.random(N_SAMPLES)
    ])


@pytest.fixture
def data2d():
    """The dataset used to build the transformer, based on a 2D-mesh."""
    tau = array(MESH2D)
    tau = tau[:, 0] - tau[:, 1]
    tau.flatten()
    return array([func(tau, theta) for theta in RNG.random(N_SAMPLES)])


def test_constructor() -> None:
    """Test constructor."""
    algo = KLSVD(MESH)
    assert algo.name == "KLSVD"
    assert algo.algo is None


def test_learn(data) -> None:
    """Test learn with the default number of components (None)."""
    algo = KLSVD(MESH)
    algo.fit(data)
    assert algo.n_components == 10


def test_learn_custom(data) -> None:
    """Test learn with a number of components different from the mesh size."""
    algo = KLSVD(MESH, 9)
    algo.fit(data)
    assert algo.n_components == 9


def test_transform(data, data2d) -> None:
    """Test transform."""
    algo = KLSVD(MESH)
    algo.fit(data)
    reduced_data = algo.transform(data)
    assert reduced_data.shape[0] == data.shape[0]
    assert reduced_data.shape[1] == algo.output_dimension
    algo = KLSVD(MESH2D)
    algo.fit(data2d)
    reduced_data = algo.transform(data2d)
    assert reduced_data.shape[0] == data2d.shape[0]
    assert reduced_data.shape[1] == algo.output_dimension


def test_inverse_transform(data, data2d) -> None:
    """Test inverse transform."""
    algo = KLSVD(MESH)
    algo.fit(data)
    coefficients = algo.transform(data)
    restored_data = algo.inverse_transform(coefficients)
    assert restored_data.shape[0] == data.shape[0]
    assert restored_data.shape[1] == data.shape[1]
    algo = KLSVD(MESH2D)
    algo.fit(data2d)
    coefficients = algo.transform(data2d)
    restored_data = algo.inverse_transform(coefficients)
    assert restored_data.shape[0] == data2d.shape[0]
    assert restored_data.shape[1] == data2d.shape[1]


def test_eigen(data) -> None:
    """Test eigen values and eigen vectors."""
    algo = KLSVD(MESH)
    algo.fit(data)
    assert algo.components.shape[0] == data.shape[1]
    assert algo.components.shape[1] == algo.n_components
    assert len(algo.eigenvalues) == algo.n_components


def test_mesh() -> None:
    """Test mesh."""
    algo = KLSVD(MESH)
    assert algo.mesh == MESH


def test_n_singular_values_default(data) -> None:
    """Check the default value of n_singular_values."""
    algo = KLSVD(MESH)
    algo.fit(data)
    assert ResourceMap.Get("KarhunenLoeveSVDAlgorithm-RandomSVDMaximumRank") == "1000"


def test_n_singular_values(data) -> None:
    """Check changing the value of n_singular_values."""
    algo = KLSVD(MESH, n_singular_values=10)
    algo.fit(data)
    assert ResourceMap.Get("KarhunenLoeveSVDAlgorithm-RandomSVDMaximumRank") == "10"


def test_use_random_svd_default(data) -> None:
    """Check the default value of use_random_svd."""
    algo = KLSVD(MESH)
    algo.fit(data)
    assert ResourceMap.Get("KarhunenLoeveSVDAlgorithm-UseRandomSVD") == "false"


@pytest.mark.parametrize("use_random_svd", [False, True])
def test_use_random_svd(data, use_random_svd) -> None:
    """Check changing use_random_svd."""
    algo = KLSVD(MESH, use_random_svd=use_random_svd)
    algo.fit(data)
    assert (
        ResourceMap.Get("KarhunenLoeveSVDAlgorithm-UseRandomSVD")
        == str(use_random_svd).lower()
    )


def test_use_halko2010_default(data) -> None:
    """Check the default value of use_halko2010."""
    algo = KLSVD(MESH)
    algo.fit(data)
    assert (
        ResourceMap.Get("KarhunenLoeveSVDAlgorithm-RandomSVDVariant").lower()
        == "halko2010"
    )


@pytest.mark.parametrize("use_halko2010", [False, True])
def test_use_halko2010(data, use_halko2010) -> None:
    """Check changing the value of use_halko2010."""
    algo = KLSVD(MESH, use_halko2010=use_halko2010)
    algo.fit(data)
    assert ResourceMap.Get("KarhunenLoeveSVDAlgorithm-RandomSVDVariant").lower() == (
        "halko2010" if use_halko2010 else "halko2011"
    )


def test_shape(data) -> None:
    """Check the shapes of the data."""
    klsvd = KLSVD(MESH)
    klsvd.fit(data)
    n, p = data.shape
    q = klsvd.n_components
    transformed_data = klsvd.transform(data)
    assert transformed_data.shape == (n, q)
    assert klsvd.inverse_transform(transformed_data).shape == (n, p)

    assert klsvd.transform(data[0]).shape == (q,)
    assert klsvd.inverse_transform(transformed_data[0]).shape == (p,)
