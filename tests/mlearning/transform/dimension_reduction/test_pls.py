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
"""Test partial least square regression."""
from __future__ import division, unicode_literals

from typing import Tuple

import pytest
from numpy import ndarray
from numpy import sum as npsum
from numpy.random import rand

from gemseo.mlearning.transform.dimension_reduction.pls import PLS

N_SAMPLES = 10
N_FEATURES = 8


@pytest.fixture
def data():  # type: (...) -> Tuple[ndarray,ndarray]
    """The dataset used to build the transformer, based on a 1D-mesh."""
    input_data = rand(N_SAMPLES, N_FEATURES)
    output_data = npsum(input_data, 1)[:, None]
    return input_data, output_data


def test_constructor():
    """Test constructor."""
    n_components = 3
    pca = PLS(n_components=n_components)
    assert pca.name == "PLS"
    assert pca.algo is not None
    assert pca.n_components == n_components


def test_learn(data):
    """Test learn."""
    input_data, output_data = data
    n_components = 3
    pca = PLS(n_components=n_components)
    pca.fit(input_data, output_data)


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
