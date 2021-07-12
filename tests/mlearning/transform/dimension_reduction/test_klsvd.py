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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Syver Doving Agdestein
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test dimension reduction with Karhunen-Loeve singular value decomposition."""
from __future__ import division, unicode_literals

import pytest
from numpy import array, linspace, pi, sin
from numpy.random import rand

from gemseo.mlearning.transform.dimension_reduction.klsvd import KLSVD

N_SAMPLES = 100

MESH_SIZE = 10
MESH = [[t_i] for t_i in linspace(0, 1, MESH_SIZE)]
MESH2D = [
    [x_i, x_j] for x_i in linspace(0, 1, MESH_SIZE) for x_j in linspace(0, 1, MESH_SIZE)
]


def func(tau, theta):
    """Data generating function."""
    return sin(2 * pi * (tau - theta)) + 1


@pytest.fixture
def data():
    """The dataset used to build the transformer, based on a 1D-mesh."""
    return array([func(array(MESH).flatten(), theta) for theta in rand(N_SAMPLES)])


@pytest.fixture
def data2d():
    """The dataset used to build the transformer, based on a 2D-mesh."""
    tau = array(MESH2D)
    tau = tau[:, 0] - tau[:, 1]
    tau.flatten()
    data_ = array([func(tau, theta) for theta in rand(N_SAMPLES)])
    return data_


def test_constructor():
    """Test constructor."""
    algo = KLSVD(MESH)
    assert algo.name == "KLSVD"
    assert algo.algo is None


def test_learn(data, data2d):
    """Test learn."""
    algo = KLSVD(MESH)
    algo.fit(data)
    assert len(algo.algo.getModes()) == 5

    algo = KLSVD(MESH2D)
    algo.fit(data2d)
    assert len(algo.algo.getModes()) == 5

    algo = KLSVD(MESH, 10)
    algo.fit(data)
    assert len(algo.algo.getModes()) == 10


def test_transform(data, data2d):
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


def test_inverse_transform(data, data2d):
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


def test_eigen(data):
    """Test eigen values and eigen vectors."""
    algo = KLSVD(MESH)
    algo.fit(data)
    assert algo.components.shape[0] == data.shape[1]
    assert algo.components.shape[1] == algo.n_components
    assert len(algo.eigenvalues) == algo.n_components


def test_mesh():
    """Test mesh."""
    algo = KLSVD(MESH)
    assert algo.mesh == MESH
