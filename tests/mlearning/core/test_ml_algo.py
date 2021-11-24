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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test machine learning algorithm module."""
from __future__ import division, unicode_literals

import pytest
from numpy import arange, array_equal

from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.kmeans import KMeans
from gemseo.mlearning.core.factory import MLAlgoFactory
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from gemseo.utils.py23_compat import Path, xrange


class NewMLAlgo(MLAlgo):
    """New machine learning algorithm class."""

    LIBRARY = "NewLibrary"

    def learn(self, samples=None):
        super(NewMLAlgo, self).learn(samples=samples)
        self._trained = True

    def _learn(self, indices):
        pass


@pytest.fixture
def dataset():  # type: (...) -> Dataset
    """The dataset used to train the machine learning algorithms."""
    data = arange(30).reshape(10, 3)
    variables = ["x_1", "x_2"]
    sizes = {"x_1": 1, "x_2": 2}
    samples = Dataset("dataset_name")
    samples.set_from_array(data, variables, sizes)
    return samples


def test_constructor(dataset):
    """Test construction."""
    ml_algo = MLAlgo(dataset)
    assert ml_algo.algo is None
    assert not ml_algo.is_trained
    kmeans = KMeans(dataset)
    kmeans.learn()
    assert kmeans.is_trained


def test_learning_samples(dataset):
    algo = NewMLAlgo(dataset)
    algo.learn()
    assert list(algo.learning_samples_indices) == list(xrange(len(dataset)))
    algo = NewMLAlgo(dataset)
    algo.learn(samples=[0, 1])
    assert algo.learning_samples_indices == [0, 1]


@pytest.mark.parametrize("samples", [xrange(10), [1, 2]])
@pytest.mark.parametrize("trained", [False, True])
def test_str(dataset, samples, trained):
    """Test string representation."""
    ml_algo = NewMLAlgo(dataset)
    ml_algo._learning_samples_indices = samples
    ml_algo._trained = trained
    expected = "\n".join(["NewMLAlgo()", "   based on the NewLibrary library"])
    if ml_algo.is_trained:
        expected += "\n   built from {} learning samples".format(len(samples))
    assert str(ml_algo) == expected


def test_scale(dataset):
    """Test scaler in MLAlgo."""
    ml_algo = MLAlgo(dataset, transformer={"parameters": MinMaxScaler()})
    assert isinstance(ml_algo.transformer["parameters"], MinMaxScaler)


def test_save_and_load(dataset, tmp_path, monkeypatch, reset_factory):
    """Test save and load."""
    # Let the factory find NewMLAlgo
    monkeypatch.setenv("GEMSEO_PATH", Path(__file__).parent)

    model = NewMLAlgo(dataset)
    model.learn()
    factory = MLAlgoFactory()

    directory_path = model.save(path=tmp_path, save_learning_set=True)
    imported_model = factory.load(directory_path)
    assert array_equal(
        imported_model.learning_set.get_data_by_names(["x_1"], False),
        model.learning_set.get_data_by_names(["x_1"], False),
    )
    assert imported_model.is_trained

    directory_path = model.save(path=tmp_path)
    imported_model = factory.load(directory_path)
    assert len(model.learning_set) == 0
    assert len(imported_model.learning_set) == 0
    assert imported_model.is_trained
    assert imported_model.sizes == dataset.sizes
