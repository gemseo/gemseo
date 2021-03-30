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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
""" Test polynomial chaos expansion regression module. """
from __future__ import absolute_import, division, unicode_literals

import pytest
from future import standard_library
from numpy import allclose, array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.dataset import Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.mlearning.regression.pce import PCERegression
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler

standard_library.install_aliases()

LEARNING_SIZE = 9


@pytest.fixture
def discipline():
    """ Discipline from R^2 to R^2. """
    expressions_dict = {"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"}
    discipline_ = AnalyticDiscipline("func", expressions_dict)
    return discipline_


@pytest.fixture
def dataset(discipline):
    """ Dataset from a R^2 -> R^2 function sampled over [0,1]^2. """

    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=0.0, u_b=1.0)
    design_space.add_variable("x_2", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("dataset_name")


@pytest.fixture
def model(dataset, prob_space):
    """ Define model from data. """
    pce = PCERegression(dataset, prob_space)
    pce.learn()
    return pce


@pytest.fixture
def prob_space():
    """ Probability space. """
    space = ParameterSpace()
    space.add_random_variable("x_1", "OTUniformDistribution")
    space.add_random_variable("x_2", "OTUniformDistribution")
    return space


def test_constructor(dataset, prob_space):
    """ Test construction."""
    model_ = PCERegression(dataset, prob_space)
    assert model_.algo is None
    model_ = PCERegression(dataset, prob_space, strategy="Quad")
    assert model_.proj_strategy is not None
    with pytest.raises(ValueError):
        PCERegression(dataset, prob_space, strategy="foo")
    prob_space.remove_variable("x_1")
    with pytest.raises(ValueError):
        PCERegression(dataset, prob_space)


def test_transform(dataset, prob_space):
    """ Test correct handling of transformers (Not supported). """
    PCERegression(dataset, prob_space, transformer={})  # Should not raise erro
    with pytest.raises(ValueError):
        PCERegression(
            dataset, prob_space, transformer={dataset.INPUT_GROUP: MinMaxScaler()}
        )


def test_learn(dataset, prob_space):
    """ Test learn."""
    model_ = PCERegression(dataset, prob_space)
    model_.learn()
    assert model_.algo is not None


def test_prediction(model):
    """ Test prediction. """
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    input_values = {"x_1": array([[1.0], [1], [1]]), "x_2": array([[1.0], [1], [1]])}
    prediction = model.predict(input_value)
    predictions = model.predict(input_values)
    assert isinstance(prediction, dict)
    assert isinstance(predictions, dict)
    assert allclose(prediction["y_1"], array([9.0]))
    assert allclose(prediction["y_2"], array([-9.0]))


def test_prediction_quad(prob_space, discipline):
    """ Test prediction. """
    dataset_ = Dataset()
    assert not dataset_
    model = PCERegression(dataset_, prob_space, strategy="Quad")
    assert dataset_
    assert model.sample.shape == (9, 2)
    model = PCERegression(dataset_, prob_space, strategy="Quad", n_quad=4)
    assert model.sample.shape == (4, 2)
    model_ = PCERegression(
        Dataset(), prob_space, discipline=discipline, strategy="Quad"
    )
    model_.learn()
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    prediction = model_.predict(input_value)
    assert isinstance(prediction, dict)
    assert allclose(prediction["y_1"], array([9.0]))
    assert allclose(prediction["y_2"], array([-9.0]))
    model_ = PCERegression(
        Dataset(), prob_space, discipline=discipline, strategy="Quad", stieltjes=False
    )
    model_.learn()
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    prediction = model_.predict(input_value)
    assert isinstance(prediction, dict)
    assert allclose(prediction["y_1"], array([9.0]))
    assert allclose(prediction["y_2"], array([-9.0]))


def test_prediction_sparse(dataset, prob_space):
    """ Test prediction. """
    model = PCERegression(dataset, prob_space, strategy="SparseLS")


def test_prediction_wrong_strategy(dataset, prob_space):
    """ Test prediction. """
    with pytest.raises(ValueError):
        PCERegression(dataset, prob_space, strategy="wrong_strategy")


def test_prediction_jacobian(model):
    """ Test jacobian prediction. """
    input_value = {"x_1": array([1.0]), "x_2": array([2.0])}
    jac = model.predict_jacobian(input_value)
    assert isinstance(jac, dict)
    assert allclose(jac["y_1"]["x_1"], array([[2.0]]))
    assert allclose(jac["y_1"]["x_2"], array([[3.0]]))
    assert allclose(jac["y_2"]["x_1"], array([[-2.0]]))
    assert allclose(jac["y_2"]["x_2"], array([[-3.0]]))


def test_sobol(dataset, prob_space):
    """ Test compute_sobol."""
    model_ = PCERegression(dataset, prob_space)
    model_.learn()
    first_order, second_order = model_.compute_sobol()
    assert isinstance(first_order, list)
    assert isinstance(second_order, list)
    assert len(first_order) == 2
    assert len(second_order) == 2
