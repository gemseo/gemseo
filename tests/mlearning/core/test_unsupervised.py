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
"""Test unsupervised machine learning algorithm module."""

from __future__ import annotations

import pytest
from numpy import arange

from gemseo.datasets.dataset import Dataset
from gemseo.mlearning.core.unsupervised import MLUnsupervisedAlgo
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.utils.testing.helpers import concretize_classes


@pytest.fixture()
def dataset() -> Dataset:
    """The dataset used to train the unsupervised machine learning algorithms."""
    data = arange(30).reshape(10, 3)
    variables = ["x_1", "x_2"]
    variable_names_to_n_components = {"x_1": 1, "x_2": 2}
    dataset_ = Dataset.from_array(data, variables, variable_names_to_n_components)
    dataset_.name = "dataset_name"
    return dataset_


def test_constructor(dataset):
    """Test construction."""
    with concretize_classes(MLUnsupervisedAlgo):
        ml_algo = MLUnsupervisedAlgo(dataset)

    assert ml_algo.algo is None
    assert ml_algo.var_names == dataset.get_variable_names(dataset.DEFAULT_GROUP)


def test_variable_limitation(dataset):
    """Test specifying learning variables."""
    with concretize_classes(MLUnsupervisedAlgo):
        ml_algo_limited = MLUnsupervisedAlgo(
            dataset,
            transformer={"x_1": MinMaxScaler(), "x_2": MinMaxScaler()},
            var_names=["x_1"],
        )

    assert ml_algo_limited.var_names == ["x_1"]
