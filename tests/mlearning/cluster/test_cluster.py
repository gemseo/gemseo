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
"""Test machine learning clustering algorithm module."""
from __future__ import absolute_import, division, unicode_literals

import pytest
from numpy import arange, zeros

from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.cluster import MLClusteringAlgo


@pytest.fixture
def dataset():
    """Build an input-output dataset."""
    data = arange(30).reshape(10, 3)
    variables = ["x_1", "x_2"]
    sizes = {"x_1": 1, "x_2": 2}
    dataset_ = Dataset("dataset_name")
    dataset_.set_from_array(data, variables, sizes)
    return dataset_


@pytest.fixture
def new_algo():
    """Create new machine learning algorithm."""

    class NewAlgo(MLClusteringAlgo):
        """New machine learning algorithm class."""

        def _fit(self, data):
            pass

        def _predict(self, data):
            pass

        def _predict_proba_soft(self, data):
            pass

    return NewAlgo


def test_notimplementederror(dataset):
    """Test not implemented methods."""
    ml_algo = MLClusteringAlgo(dataset)
    with pytest.raises(NotImplementedError):
        ml_algo.learn()
    with pytest.raises(NotImplementedError):
        ml_algo.predict({"x_1": zeros(1), "x_2": zeros(2)})
    with pytest.raises(NotImplementedError):
        ml_algo.predict_proba({"x_1": zeros(1), "x_2": zeros(2)})
    with pytest.raises(NotImplementedError):
        ml_algo.predict_proba({"x_1": zeros(1), "x_2": zeros(2)}, hard=False)


def test_labels(dataset, new_algo):
    """Test clustering labels."""
    algo = new_algo(dataset)
    with pytest.raises(ValueError):
        algo.learn()
