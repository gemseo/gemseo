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
from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.kmeans import KMeans
from gemseo.mlearning.core.factory import MLAlgoFactory
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.transform.scaler.scaler import Scaler
from gemseo.utils.pytest_conftest import concretize_classes
from numpy import arange
from numpy import array
from numpy import array_equal

from .new_ml_algo.new_ml_algo import NewMLAlgo


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the machine learning algorithms."""
    data = arange(30).reshape(10, 3)
    variables = ["x_1", "x_2"]
    sizes = {"x_1": 1, "x_2": 2}
    samples = Dataset("dataset_name")
    samples.set_from_array(data, variables, sizes)
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

    directory_path = model.save(save_learning_set=True)
    imported_model = factory.load(directory_path)
    assert array_equal(
        imported_model.learning_set.get_data_by_names(["x_1"], False),
        model.learning_set.get_data_by_names(["x_1"], False),
    )
    assert imported_model.is_trained

    directory_path = model.save()
    imported_model = factory.load(directory_path)
    assert len(model.learning_set) == 0
    assert len(imported_model.learning_set) == 0
    assert imported_model.is_trained
    assert imported_model.sizes == dataset.sizes


def test_transformers_error(dataset):
    """Check that MLAlgo cannot use a transformer for both group and variable."""
    dataset = Dataset()
    dataset.add_variable("x", array([[1.0]]), group="foo")
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
