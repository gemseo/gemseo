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
"""Test machine learning model module."""

from __future__ import annotations

import pytest
from numpy import arange
from numpy import array

from gemseo.datasets.io_dataset import IODataset
from gemseo.machine_learning.clustering.models.kmeans import KMeans
from gemseo.machine_learning.core.models.ml_model import BaseMLModel
from gemseo.machine_learning.core.models.ml_model_settings import BaseMLModelSettings
from gemseo.machine_learning.transformers.scaler.scaler import Scaler
from gemseo.utils.repr_html import REPR_HTML_WRAPPER
from gemseo.utils.testing.helpers import assert_exception
from gemseo.utils.testing.helpers import concretize_classes

from .new_ml_model.new_ml_model import NewMLModel


class DummyMLModel(BaseMLModel):
    settings_class = BaseMLModelSettings


@pytest.fixture
def dataset() -> IODataset:
    """The dataset used to train the machine learning models."""
    data = arange(30).reshape(10, 3)
    variables = ["x_1", "x_2"]
    variable_name_to_n_components = {"x_1": 1, "x_2": 2}
    samples = IODataset.from_array(data, variables, variable_name_to_n_components)
    samples.name = "dataset_name"
    return samples


def test_constructor(dataset) -> None:
    """Test construction."""
    with concretize_classes(DummyMLModel):
        ml_model = DummyMLModel(dataset)

    assert ml_model.algo is None
    assert not ml_model.is_trained
    kmeans = KMeans(dataset)
    kmeans.learn()
    assert kmeans.is_trained


@pytest.mark.parametrize(
    ("kwargs", "expected"), [({}, list(range(10))), ({"samples": [0, 1]}, [0, 1])]
)
def test_learning_samples_indices(dataset, kwargs, expected) -> None:
    model = NewMLModel(dataset)
    assert model.learning_samples_indices == list(range(10))
    model.learn(**kwargs)
    assert model.learning_samples_indices == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, ["a", "b", "c"]), ({"samples": ["c", "a"]}, ["c", "a"])],
)
def test_learning_samples_indices_with_abc_indices(kwargs, expected) -> None:
    dataset = IODataset.from_array(arange(3).reshape(3, 1))
    names = ["a", "b", "c"]
    dataset.index = names
    model = NewMLModel(dataset)
    assert model.learning_samples_indices == names
    model.learn(**kwargs)
    assert model.learning_samples_indices == expected


@pytest.mark.parametrize("samples", [range(10), [1, 2]])
@pytest.mark.parametrize("trained", [False, True])
def test_repr_str(dataset, samples, trained) -> None:
    """Test string representations."""
    ml_model = NewMLModel(dataset)
    ml_model._learning_samples_indices = samples
    ml_model._trained = trained
    expected = (
        "NewMLModel(parameters={}, transformer={})\n   based on the NewLibrary library"
    )
    if ml_model.is_trained:
        expected += f"\n   built from {len(samples)} learning samples"

    assert repr(ml_model) == str(ml_model) == expected
    assert str(ml_model) == expected


def test_repr_html(dataset) -> None:
    """Check the HTML representation of an ML model."""
    assert NewMLModel(dataset)._repr_html_() == REPR_HTML_WRAPPER.format(
        "NewMLModel(parameters={}, transformer={})<br/>"
        "<ul><li>based on the NewLibrary library</li></ul>"
    )


@pytest.mark.parametrize(
    "transformer", ["Scaler", ("Scaler", {"offset": 2.0}), Scaler()]
)
def test_transformer(dataset, transformer) -> None:
    """Check if transformers are correctly passed."""
    with concretize_classes(DummyMLModel):
        ml_model = DummyMLModel(
            dataset, BaseMLModelSettings(transformer={"parameters": transformer})
        )

    assert isinstance(ml_model.transformer["parameters"], Scaler)
    if isinstance(transformer, tuple):
        assert ml_model.transformer["parameters"].offset == 2.0


def test_transformer_wrong_type(dataset, snapshot) -> None:
    """Check that using a wrong transformer type raises a ValueError."""
    with (
        assert_exception(ValueError, snapshot),
        concretize_classes(DummyMLModel),
    ):
        DummyMLModel(dataset, BaseMLModelSettings(transformer={"parameters": 1}))


def test_transformers_error(dataset, snapshot) -> None:
    """Check that BaseMLModel cannot use a transformer for both group and variable."""
    dataset = IODataset()
    dataset.add_variable("x", array([[1.0]]), group_name="foo")
    with (
        assert_exception(ValueError, snapshot),
        concretize_classes(DummyMLModel),
    ):
        DummyMLModel(
            dataset,
            BaseMLModelSettings(
                transformer={"x": "MinMaxScaler", "foo": "MinMaxScaler"}
            ),
        )
