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
#                           documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Unit test for MLAlgoFactory class in gemseo.mlearning.core.factory."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.factory import MLAlgoFactory
from numpy import arange

LEARNING_SIZE = 9


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the machine learning algorithms."""
    data = arange(30).reshape((10, 3))
    dataset_ = Dataset()
    dataset_.add_group(Dataset.INPUT_GROUP, data[:, :2])
    dataset_.add_group(Dataset.OUTPUT_GROUP, data[:, [2]])
    return dataset_


def test_constructor():
    """Test factory constructor."""
    assert {
        "GaussianMixture",
        "GaussianProcessRegressor",
        "KMeans",
        "KNNClassifier",
        "LinearRegressor",
        "MOERegressor",
        "PCERegressor",
        "PolynomialRegressor",
        "RBFRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
        "SVMClassifier",
    } <= set(MLAlgoFactory().models)


def test_create(dataset):
    """Test the creation of a model from data."""
    factory = MLAlgoFactory()
    ml_algo = factory.create("LinearRegressor", data=dataset)
    assert hasattr(ml_algo, "parameters")


def test_load(dataset, tmp_wd):
    """Test the loading of a model from data."""
    factory = MLAlgoFactory()
    ml_algo = factory.create("RandomForestClassifier", data=dataset)
    dirname = ml_algo.save()
    loaded_ml_algo = factory.load(dirname)
    assert hasattr(loaded_ml_algo, "parameters")


def test_available_models():
    """Test the getter of available regression models."""
    factory = MLAlgoFactory()
    assert "KMeans" in factory.models


def test_is_available():
    """Test the existence of a regression model."""
    factory = MLAlgoFactory()
    assert factory.is_available("PolynomialRegressor")
    assert not factory.is_available("Dummy")
