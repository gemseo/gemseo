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
""" Test principal component analysis dimension reduction. """
from __future__ import absolute_import, division, unicode_literals

import pytest
from future import standard_library
from numpy import linspace

from gemseo.mlearning.transform.dimension_reduction.kpca import KPCA

standard_library.install_aliases()

N_SAMPLES = 10
N_FEATURES = 8


@pytest.fixture
def data():
    """ Build an input-output dataset. """
    data_ = linspace(0, 1, N_SAMPLES * N_FEATURES)
    return data_.reshape(N_SAMPLES, N_FEATURES)


def test_constructor():
    """ Test constructor. """
    n_components = 3
    kpca = KPCA(n_components=n_components)
    assert kpca.name == "KPCA"
    assert kpca.algo is not None
    assert kpca.n_components == n_components


def test_learn(data):
    """ Test learn. """
    n_components = 3
    kpca = KPCA(n_components=n_components)
    kpca.fit(data)


def test_transform(data):
    """ Test transform. """
    n_components = 3
    kpca = KPCA(n_components=n_components)
    kpca.fit(data)
    reduced_data = kpca.transform(data)
    assert reduced_data.shape[0] == data.shape[0]
    assert reduced_data.shape[1] == n_components


def test_inverse_transform(data):
    """ Test inverse transform. """
    n_components = 3
    kpca = KPCA(n_components=n_components)
    kpca.fit(data)
    data = linspace(0, 1, N_SAMPLES * n_components)
    data = data.reshape((N_SAMPLES, n_components))
    restored_data = kpca.inverse_transform(data)
    assert restored_data.shape[0] == data.shape[0]
    assert restored_data.shape[1] == N_FEATURES
