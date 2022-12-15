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
#        :author: Syver Doving Agdestein, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test polynomial regression module."""
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.core.dataset import Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.api import import_regression_model
from gemseo.mlearning.regression.polyreg import PolynomialRegressor
from numpy import allclose
from numpy import array
from numpy import hstack
from numpy import linspace
from numpy import meshgrid
from numpy import sqrt
from numpy import zeros
from scipy.special import comb

LEARNING_SIZE = 50
DEGREE = 5
N_INPUTS = 2
N_OUTPUTS = 3
N_POWERS = comb(N_INPUTS + DEGREE, N_INPUTS, exact=True) - 1

COEFFICIENTS = zeros((N_OUTPUTS, N_POWERS))
COEFFICIENTS[0, [0, 4]] = [1, 1]
COEFFICIENTS[1, [3, 5]] = [4, 5]
COEFFICIENTS[2, [7, 19]] = [10, 7]

# 1D
INPUT_VALUE = {"x_1": array([1]), "x_2": array([2])}

# 2D
ANOTHER_INPUT_VALUE = {
    "x_1": array([[0], [0], [1], [2]]),
    "x_2": array([[0], [1], [2], [2]]),
}


@pytest.fixture
def dataset():
    """Dataset from a R^2 -> R^3 function sampled over [-1, 2]^2."""
    root_learning_size = int(sqrt(LEARNING_SIZE))
    x_1 = linspace(-1, 2, root_learning_size)
    x_2 = linspace(-1, 2, root_learning_size)
    x_1, x_2 = meshgrid(x_1, x_2)
    x_1, x_2 = x_1.flatten()[:, None], x_2.flatten()[:, None]
    y_1 = 1 + x_1 + x_2**2
    y_2 = 3 + 4 * x_1 * x_2 + 5 * x_1**3
    y_3 = 10 * x_1 * x_2**2 + 7 * x_2**5

    data = hstack([x_1, x_2, y_1, y_2, y_3])
    variables = ["x_1", "x_2", "y_1", "y_2", "y_3"]
    sizes = {"x_1": 1, "x_2": 1, "y_1": 1, "y_2": 1, "y_3": 1}
    groups = {
        "x_1": Dataset.INPUT_GROUP,
        "x_2": Dataset.INPUT_GROUP,
        "y_1": Dataset.OUTPUT_GROUP,
        "y_2": Dataset.OUTPUT_GROUP,
        "y_3": Dataset.OUTPUT_GROUP,
    }

    dataset_ = Dataset()
    dataset_.set_from_array(data, variables, sizes, groups)
    return dataset_


@pytest.fixture
def dataset_from_cache() -> Dataset:
    """The dataset used to train the regression algorithms."""
    discipline = AnalyticDiscipline(
        {
            "y_1": "1 + x_1 + x_2**2",
            "y_2": "3 + 4*x_1*x_2 + 5*x_1**3",
            "y_3": "10*x_1*x_2**2 + 7*x_2**5",
        }
    )
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_2", l_b=-1, u_b=2)
    design_space.add_variable("x_1", l_b=-1, u_b=2)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": LEARNING_SIZE})
    data = discipline.cache.export_to_dataset("dataset_name")
    return data


@pytest.fixture
def model(dataset) -> PolynomialRegressor:
    """A trained PolynomialRegressor."""
    polyreg = PolynomialRegressor(dataset, degree=DEGREE)
    polyreg.learn()
    return polyreg


@pytest.fixture
def model_without_intercept(dataset) -> PolynomialRegressor:
    """A trained PolynomialRegressor without intercept fitting."""
    polyreg = PolynomialRegressor(dataset, degree=DEGREE, fit_intercept=False)
    polyreg.learn()
    return polyreg


def test_constructor(dataset):
    model_ = PolynomialRegressor(dataset, degree=2)
    assert model_.algo is not None


def test_degree(dataset):
    """Test correct handling of incorrect degree ( < 1)."""
    with pytest.raises(ValueError):
        PolynomialRegressor(dataset, degree=0)


def test_learn(dataset):
    """Test learn."""
    model_ = PolynomialRegressor(dataset, degree=2)
    model_.learn()
    assert model_.algo is not None


def test_get_coefficients(model):
    """Verify that an error is raised when getting coefficients as a dictionary."""
    with pytest.raises(
        NotImplementedError,
        match=(
            "For now the coefficients can only be obtained "
            "in the form of a NumPy array"
        ),
    ):
        model.get_coefficients(as_dict=True)


def test_intercept(model, model_without_intercept):
    """Test intercept parameter from LinearRegressor class.

    Should be 0.0, as fit_intercept is False (replaced by include_bias).
    """
    assert allclose(model.intercept, array([1, 3, 0]))
    assert allclose(model_without_intercept.intercept, array([0, 0, 0]))


def test_coefficients(model):
    """Test coefficients."""
    assert model.coefficients.shape == (N_OUTPUTS, N_POWERS)
    coefficients = model.get_coefficients(as_dict=False)
    assert allclose(coefficients, COEFFICIENTS, atol=1.0e-12)


def test_prediction(model):
    """Test prediction."""
    prediction = model.predict(INPUT_VALUE)
    another_prediction = model.predict(ANOTHER_INPUT_VALUE)
    assert isinstance(prediction, dict)
    assert isinstance(another_prediction, dict)
    assert allclose(prediction["y_1"], array([6]))
    assert allclose(prediction["y_2"], array([16]))
    assert allclose(prediction["y_3"], array([264]))
    assert allclose(another_prediction["y_1"], array([[1], [2], [6], [7]]))


def test_prediction_jacobian(model):
    """Test jacobian prediction."""
    jacobian = model.predict_jacobian(INPUT_VALUE)
    another_jacobian = model.predict_jacobian(ANOTHER_INPUT_VALUE)
    assert isinstance(jacobian, dict)
    assert isinstance(another_jacobian, dict)
    assert allclose(jacobian["y_1"]["x_1"], 1)
    assert allclose(jacobian["y_1"]["x_2"], 4)
    assert allclose(jacobian["y_2"]["x_1"], 23)
    assert allclose(jacobian["y_2"]["x_2"], 4)
    assert allclose(jacobian["y_3"]["x_1"], 40)
    assert allclose(jacobian["y_3"]["x_2"], 600)
    assert allclose(another_jacobian["y_1"]["x_2"], array([0, 2, 4, 4])[:, None, None])


def test_jacobian_constant(dataset):
    """Test Jacobians linear polynomials."""
    model_ = PolynomialRegressor(dataset, degree=1)
    model_.learn()
    model_.predict_jacobian(INPUT_VALUE)
    model_.predict_jacobian(ANOTHER_INPUT_VALUE)


def test_save_and_load(model, tmp_wd):
    """Test save and load."""
    dirname = model.save()
    imported_model = import_regression_model(dirname)
    out1 = model.predict(INPUT_VALUE)
    out2 = imported_model.predict(INPUT_VALUE)
    for name, value in out1.items():
        assert allclose(value, out2[name], 1e-3)
