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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Unit test for ClassificationModelFactory class in
gemseo.mlearning.classification.factory."""
from __future__ import annotations

import pytest
from gemseo.mlearning.classification.factory import ClassificationModelFactory
from gemseo.problems.dataset.iris import IrisDataset


@pytest.fixture
def dataset() -> IrisDataset:
    """The Iris dataset used to train the classification algorithms."""
    iris = IrisDataset(as_io=True)
    return iris


def test_constructor():
    """Test factory constructor."""
    factory = ClassificationModelFactory()
    # plugins may add classes
    assert set(factory.models) <= {
        "KNNClassifier",
        "RandomForestClassifier",
        "SVMClassifier",
    }


def test_create(dataset):
    """Test the creation of a model from data."""
    factory = ClassificationModelFactory()
    knn = factory.create("KNNClassifier", data=dataset)
    assert hasattr(knn, "parameters")


def test_load(dataset, tmp_wd):
    """Test the loading of a model from data."""
    factory = ClassificationModelFactory()
    knn = factory.create("KNNClassifier", data=dataset)
    knn.learn()
    dirname = knn.save()
    loaded_knn = factory.load(dirname)
    assert hasattr(loaded_knn, "parameters")


def test_available_models():
    """Test the getter of available classification models."""
    factory = ClassificationModelFactory()
    assert "KNNClassifier" in factory.models


def test_is_available():
    """Test the existence of a classification model."""
    factory = ClassificationModelFactory()
    assert factory.is_available("KNNClassifier")
    assert not factory.is_available("Dummy")
