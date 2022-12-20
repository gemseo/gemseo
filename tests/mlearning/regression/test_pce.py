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
"""Test polynomial chaos expansion regression module."""
from __future__ import annotations

import re
from copy import deepcopy

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.dataset import Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.pce import PCERegressor
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from numpy import allclose
from numpy import array
from numpy.testing import assert_equal
from openturns import FunctionalChaosRandomVector

LEARNING_SIZE = 9


@pytest.fixture
def discipline() -> AnalyticDiscipline:
    """Discipline from R^2 to R^2."""
    return AnalyticDiscipline({"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"})


@pytest.fixture
def dataset(discipline) -> Dataset:
    """The dataset used to train the regression algorithms."""
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("dataset_name")


@pytest.fixture
def model(dataset, prob_space) -> PCERegressor:
    """A trained PCERegressor."""
    pce = PCERegressor(dataset, prob_space)
    pce.learn()
    return pce


@pytest.fixture
def untrained_model(dataset, prob_space) -> PCERegressor:
    """An untrained PCERegressor."""
    return PCERegressor(dataset, prob_space)


@pytest.fixture
def prob_space() -> ParameterSpace:
    """A probability space describing the uncertain variables."""
    space = ParameterSpace()
    space.add_random_variable("x_1", "OTUniformDistribution")
    space.add_random_variable("x_2", "OTUniformDistribution")
    return space


def test_constructor(dataset, prob_space):
    """Test construction."""
    model_ = PCERegressor(dataset, prob_space)
    assert model_.algo is None
    assert model_.SHORT_ALGO_NAME == "PCE"
    assert model_.LIBRARY == "OpenTURNS"

    model_ = PCERegressor(dataset, prob_space, strategy="Quad")
    assert model_._proj_strategy is not None
    with pytest.raises(
        ValueError,
        match=(
            "The strategy foo is not available; "
            "available ones are: LS, Quad, SparseLS."
        ),
    ):
        PCERegressor(
            dataset,
            prob_space,
            strategy="foo",
        )
    prob_space.remove_variable("x_1")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The input names ['x_1', 'x_2'] "
            "and the names of the variables of the probability space ['x_2'] "
            "are not all the same."
        ),
    ):
        PCERegressor(dataset, prob_space)


def test_transform(dataset, prob_space):
    """Test correct handling of transformers (Not supported)."""
    PCERegressor(dataset, prob_space, transformer={})  # Should not raise error
    PCERegressor(
        dataset, prob_space, transformer={dataset.OUTPUT_GROUP: MinMaxScaler()}
    )  # Should not raise error
    with pytest.raises(
        ValueError, match="PCERegressor does not support input transformers."
    ):
        PCERegressor(
            dataset, prob_space, transformer={dataset.INPUT_GROUP: MinMaxScaler()}
        )


def test_learn(dataset, prob_space):
    """Test learn."""
    model_ = PCERegressor(dataset, prob_space)
    model_.learn()
    assert model_.algo is not None


def test_prediction(model):
    """Test prediction."""
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    input_values = {"x_1": array([[1.0], [1], [1]]), "x_2": array([[1.0], [1], [1]])}
    prediction = model.predict(input_value)
    predictions = model.predict(input_values)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    assert allclose(prediction["y_1"], array([9.0]))
    assert allclose(prediction["y_2"], array([-9.0]))


@pytest.mark.parametrize("after", [False, True])
def test_deepcopy(dataset, prob_space, after):
    """Check that a model can be deepcopied before or after learning."""
    model = PCERegressor(dataset, prob_space)
    if after:
        model.learn()
    model_copy = deepcopy(model)
    if not after:
        model.learn()
        model_copy.learn()

    input_data = {"x_1": array([1.0]), "x_2": array([2.0])}
    assert_equal(model.predict(input_data), model_copy.predict(input_data))


def test_prediction_quad(prob_space, discipline):
    """Test prediction."""
    dataset_ = Dataset()
    assert not dataset_
    model = PCERegressor(dataset_, prob_space, strategy="Quad")
    assert dataset_
    assert model._sample.shape == (9, 2)
    model = PCERegressor(dataset_, prob_space, strategy="Quad", n_quad=4)
    assert model._sample.shape == (4, 2)
    model_ = PCERegressor(Dataset(), prob_space, discipline=discipline, strategy="Quad")
    model_.learn()
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    prediction = model_.predict(input_value)
    assert isinstance(prediction, dict)
    assert allclose(prediction["y_1"], array([9.0]))
    assert allclose(prediction["y_2"], array([-9.0]))
    model_ = PCERegressor(
        Dataset(), prob_space, discipline=discipline, strategy="Quad", stieltjes=False
    )
    model_.learn()
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    prediction = model_.predict(input_value)
    assert isinstance(prediction, dict)
    assert allclose(prediction["y_1"], array([9.0]))
    assert allclose(prediction["y_2"], array([-9.0]))


def test_prediction_sparse_ls(dataset, prob_space):
    """Test prediction."""
    model = PCERegressor(dataset, prob_space, strategy="SparseLS")
    model.learn()
    assert model._strategy == model.SPARSE_STRATEGY


def test_prediction_wrong_strategy(dataset, prob_space):
    """Test prediction."""
    with pytest.raises(
        ValueError,
        match=(
            "The strategy wrong_strategy is not available; "
            "available ones are: LS, Quad, SparseLS."
        ),
    ):
        PCERegressor(dataset, prob_space, strategy="wrong_strategy")


def test_prediction_jacobian(model):
    """Test jacobian prediction."""
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    jac = model.predict_jacobian(input_value)
    assert isinstance(jac, dict)
    assert allclose(jac["y_1"]["x_1"], array([[2.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[3.0]]))
    assert allclose(jac["y_2"]["x_1"], array([[-2.0]]))
    assert allclose(jac["y_2"]["x_2"], array([[-3.0]]))


def test_sobol(dataset, prob_space):
    """Test compute_sobol."""
    model_ = PCERegressor(dataset, prob_space)
    model_.learn()
    assert isinstance(model_.first_sobol_indices, dict)
    assert isinstance(model_.total_sobol_indices, dict)
    assert len(model_.first_sobol_indices) == 2
    assert len(model_.total_sobol_indices) == 2


def test_mean_cov_var_std(model):
    """Check the mean, covariance, variance and standard deviation."""
    vector = FunctionalChaosRandomVector(model.algo)
    mean = model.mean
    assert mean.shape == (2,)
    assert_equal(mean, array(vector.getMean()))

    covariance = model.covariance
    assert covariance.shape == (2, 2)
    assert_equal(covariance, array(vector.getCovariance()))

    variance = model.variance
    assert variance.shape == (2,)
    assert_equal(variance, covariance.diagonal())

    standard_deviation = model.standard_deviation
    assert standard_deviation.shape == (2,)
    assert_equal(standard_deviation, variance**0.5)


@pytest.mark.parametrize(
    "name",
    [
        "mean",
        "covariance",
        "variance",
        "standard_deviation",
        "first_sobol_indices",
        "total_sobol_indices",
    ],
)
def test_check_is_trained(untrained_model, name):
    """Check that a RuntimeError is raised when accessing properties before training."""
    with pytest.raises(
        RuntimeError,
        match=re.escape(f"The PCERegressor must be trained to access {name}."),
    ):
        getattr(untrained_model, name)


def test_ot_distribution(dataset):
    """Check that PCERegressor handles only the OTDistribution instances."""
    probability_space = ParameterSpace()
    probability_space.add_random_variable("x_1", "SPUniformDistribution")
    probability_space.add_random_variable("x_2", "SPUniformDistribution")
    with pytest.raises(
        ValueError,
        match="The probability distributions must be instances of OTDistribution.",
    ):
        PCERegressor(dataset, probability_space)
