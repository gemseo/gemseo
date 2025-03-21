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

from __future__ import annotations

import re

import pytest
from numpy import arange
from numpy import array
from numpy.testing import assert_allclose

from gemseo import from_pickle
from gemseo import to_pickle
from gemseo.datasets.dataset import Dataset
from gemseo.mlearning.clustering.algos.base_clusterer import BaseClusterer
from gemseo.mlearning.clustering.algos.factory import ClustererFactory
from gemseo.problems.dataset.iris import create_iris_dataset

FACTORY = ClustererFactory()
INPUT_VALUE = array([1.5, 1.5, 1.5, 1.5])


class NewAlgo(BaseClusterer):
    """New machine learning algorithm class."""

    def _fit(self, data) -> None:
        pass


def test_labels() -> None:
    """Test clustering labels."""
    dataset = Dataset.from_array(
        arange(30).reshape(10, 3), ["x_1", "x_2"], {"x_1": 1, "x_2": 2}
    )
    algo = NewAlgo(dataset)
    with pytest.raises(
        NotImplementedError,
        match=re.escape("NewAlgo._fit() did not set the labels attribute."),
    ):
        algo.learn()


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """The Iris dataset."""
    return create_iris_dataset()


@pytest.mark.parametrize("class_name", FACTORY.class_names)
@pytest.mark.parametrize("before_training", [False, True])
def test_pickle(class_name, dataset, before_training, tmp_wd):
    """Check that clustering models are picklable."""
    reference_model = FACTORY.create(
        class_name,
        dataset,
        var_names=("sepal_length", "sepal_width", "petal_length", "petal_width"),
    )

    if before_training:
        to_pickle(reference_model, "model.pkl")
        reference_model.learn()
    else:
        reference_model.learn()
        to_pickle(reference_model, "model.pkl")

    reference_prediction = reference_model.predict(INPUT_VALUE)

    model = from_pickle("model.pkl")
    if before_training:
        model.learn()

    output_value = model.predict(INPUT_VALUE)

    assert_allclose(output_value, reference_prediction)
