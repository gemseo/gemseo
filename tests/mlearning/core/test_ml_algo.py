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
from __future__ import annotations

import re
from pathlib import Path

import pytest
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.clustering.kmeans import KMeans
from gemseo.mlearning.core.factory import MLAlgoFactory
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.transformers.scaler.scaler import Scaler
from gemseo.utils.testing.helpers import concretize_classes
from numpy import arange
from numpy import array

from .new_ml_algo.new_ml_algo import NewMLAlgo


@pytest.fixture
def dataset() -> IODataset:
    """The dataset used to train the machine learning algorithms."""
    data = arange(30).reshape(10, 3)
    variables = ["x_1", "x_2"]
    variable_names_to_n_components = {"x_1": 1, "x_2": 2}
    samples = IODataset.from_array(data, variables, variable_names_to_n_components)
    samples.name = "dataset_name"
    return samples


def test_constructor(dataset):
    """Test construction."""
    with concretize_classes(MLAlgo):
        ml_algo = MLAlgo(dataset)

    assert ml_algo.algo is None
    assert not ml_algo.is_trained
    kmeans = KMeans(dataset)
    kmeans.learn()
    assert kmeans.is_trained


def test_learning_samples(dataset):
    algo = NewMLAlgo(dataset)
    algo.learn()
    assert list(algo.learning_samples_indices) == list(range(len(dataset)))
    algo = NewMLAlgo(dataset)
    algo.learn(samples=[0, 1])
    assert algo.learning_samples_indices == [0, 1]


@pytest.mark.parametrize("samples", [range(10), [1, 2]])
@pytest.mark.parametrize("trained", [False, True])
def test_str(dataset, samples, trained):
    """Test string representation."""
    ml_algo = NewMLAlgo(dataset)
    ml_algo._learning_samples_indices = samples
    ml_algo._trained = trained
    expected = "\n".join(["NewMLAlgo()", "   based on the NewLibrary library"])
    if ml_algo.is_trained:
        expected += f"\n   built from {len(samples)} learning samples"
    assert str(ml_algo) == expected


@pytest.mark.parametrize(
    "transformer", ["Scaler", ("Scaler", {"offset": 2.0}), Scaler()]
)
def test_transformer(dataset, transformer):
    """Check if transformers are correctly passed."""
    with concretize_classes(MLAlgo):
        ml_algo = MLAlgo(dataset, transformer={"parameters": transformer})

    assert isinstance(ml_algo.transformer["parameters"], Scaler)
    if isinstance(transformer, tuple):
        assert ml_algo.transformer["parameters"].offset == 2.0


def test_transformer_wrong_type(dataset):
    """Check that using a wrong transformer type raises a ValueError."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Transformer type must be "
            "either Transformer, Tuple[str, Mapping[str, Any]] or str."
        ),
    ):
        with concretize_classes(MLAlgo):
            MLAlgo(dataset, transformer={"parameters": 1})


def test_save_and_load(dataset, tmp_wd, monkeypatch, reset_factory):
    """Test save and load."""
    # Let the factory find NewMLAlgo
    monkeypatch.setenv("GEMSEO_PATH", Path(__file__).parent / "new_ml_algo")

    model = NewMLAlgo(dataset)
    model.learn()
    factory = MLAlgoFactory()

    directory_path = model.to_pickle(save_learning_set=True)
    imported_model = factory.load(directory_path)
    assert imported_model.learning_set.get_view(variable_names="x_1").equals(
        model.learning_set.get_view(variable_names="x_1")
    )
    assert imported_model.is_trained

    directory_path = model.to_pickle()
    imported_model = factory.load(directory_path)
    assert len(model.learning_set) == 10
    assert len(imported_model.learning_set) == 10
    assert imported_model.is_trained
    assert imported_model.sizes == dataset.variable_names_to_n_components


def test_transformers_error(dataset):
    """Check that MLAlgo cannot use a transformer for both group and variable."""
    dataset = IODataset()
    dataset.add_variable("x", array([[1.0]]), group_name="foo")
    with pytest.raises(
        ValueError,
        match=(
            "An MLAlgo cannot have both a transformer "
            "for all variables of a group and a transformer "
            "for one variable of this group."
        ),
    ):
        with concretize_classes(MLAlgo):
            MLAlgo(dataset, transformer={"x": "MinMaxScaler", "foo": "MinMaxScaler"})
