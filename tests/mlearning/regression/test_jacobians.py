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
#        :author: Syver Doeving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test analytical Jacobian expressions against finite difference approximations.

This is done using the built in check method of Discipline. The regression models are
thus converted to surrogate disciplines. The Jacobians are checked over different
combinations of datasets (scalar and vector inputs and outputs), transformers and
parameters.
"""

from __future__ import annotations

import re
from operator import itemgetter
from typing import TYPE_CHECKING

import pytest
from numpy import arange
from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.surrogate import SurrogateDiscipline
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.rbf_settings import RBF
from gemseo.mlearning.transformers.dimension_reduction.pca import PCA
from gemseo.mlearning.transformers.scaler.scaler import Scaler
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.utils.testing.helpers import concretize_classes

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset

LEARNING_SIZE = 10


def create_dataset(
    dataset_name, expressions, design_space_variables, objective_name
) -> IODataset:
    """Return a dataset from a sampled function.

    Args:
        dataset_name (str): The name of the dataset.
        expressions (dict[str,str]): The expressions to be sampled.
        design_space_variables (dict[str,dict[str,str]]): A map from design space
            variables names to bounds to be passed to DesignSpace.add_variable.
        objective_name (str): The name of the objective variable.
    """
    discipline = AnalyticDiscipline(expressions)
    discipline.set_cache(discipline.CacheType.MEMORY_FULL)
    design_space = DesignSpace()
    design_space.add_variable("x_1", lower_bound=-3.0, upper_bound=3.0)
    for name, bounds in design_space_variables.items():
        design_space.add_variable(name, **bounds)
    scenario = DOEScenario(
        [discipline], objective_name, design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.execute(algo_name="LHS", n_samples=LEARNING_SIZE)
    return discipline.cache.to_dataset(dataset_name)


# the following contains the arguments passed to create_dataset
DATASETS_DESCRIPTIONS = (
    # Dataset from a R -> R function sampled over [0,1]^2
    ("scalar_scalar", {"y_1": "1+3*x_1"}, {}, "y_1"),
    # Dataset from a R -> R^3 function sampled over [0,1]^2
    (
        "scalar_vector",
        {"y_1": "1+2*x_1", "y_2": "-1-2*x_1", "y_3": "3"},
        {},
        "y_2",
    ),
    # Dataset from a R^2 -> R function sampled over [0,1]^2
    (
        "vector_scalar",
        {"y_1": "1+2*x_1+3*x_2"},
        {"x_2": {"lower_bound": -3.0, "upper_bound": 3.0}},
        "y_1",
    ),
    # Dataset from a R^2 -> R^3 function sampled over [0,1]^2
    (
        "linear",
        {"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2", "y_3": "3"},
        {"x_2": {"lower_bound": -3.0, "upper_bound": 3.0}},
        "y_1",
    ),
    # Dataset from a R^3 -> R^3 function sampled over [0,1]^2
    (
        "polynomial",
        {
            "y_1": "1+2*x_1+3*x_2 + 0.5*x_1**5 + 4*x_1*x_2**3",
            "y_2": "-1-2*x_1-3*x_2 - 0.5*x_2**4+ 7*x_1**3*x_3**2",
            "y_3": "3-9*x_3**2",
        },
        {
            "x_2": {"lower_bound": -3.0, "upper_bound": 3.0},
            "x_3": {"lower_bound": -4.0, "upper_bound": 4.0},
        },
        "y_1",
    ),
)

TRANSFORMERS = (
    {},
    {IODataset.INPUT_GROUP: Scaler(offset=5, coefficient=3)},
    {IODataset.OUTPUT_GROUP: Scaler(offset=-3, coefficient=0.1)},
    {
        IODataset.INPUT_GROUP: Scaler(offset=10, coefficient=4),
        IODataset.OUTPUT_GROUP: Scaler(offset=-7, coefficient=-8),
    },
    {
        IODataset.INPUT_GROUP: PCA(n_components=1),
        IODataset.OUTPUT_GROUP: PCA(n_components=1),
    },
)


@pytest.fixture(
    scope="module",
    params=DATASETS_DESCRIPTIONS,
    ids=map(itemgetter(0), DATASETS_DESCRIPTIONS),
)
def dataset(request) -> Dataset:
    """Return one dataset by one at runtime from DATASETS_DESCRIPTIONS."""
    return create_dataset(*request.param)


def test_regression_model() -> None:
    """Test that by default the computation of the Jacobian raises an error."""
    dataset = create_dataset(*DATASETS_DESCRIPTIONS[0])
    with (
        pytest.raises(
            NotImplementedError,
            match=re.escape("Derivatives are not available for BaseRegressor."),
        ),
        concretize_classes(BaseRegressor),
    ):
        BaseRegressor(dataset).predict_jacobian(array([1.0]))


@pytest.mark.parametrize("transformer", TRANSFORMERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_linreg(dataset, transformer, fit_intercept) -> None:
    """Test linear regression Jacobians."""
    discipline = SurrogateDiscipline(
        "LinearRegressor",
        data=dataset,
        transformer=transformer,
        fit_intercept=fit_intercept,
    )
    discipline.check_jacobian()


@pytest.mark.parametrize("transformer", TRANSFORMERS)
@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("degree", arange(1, 5))
def test_polyreg(dataset, transformer, fit_intercept, degree) -> None:
    """Test polynomial regression Jacobians."""
    discipline = SurrogateDiscipline(
        "PolynomialRegressor",
        data=dataset,
        transformer=transformer,
        fit_intercept=fit_intercept,
        degree=degree,
    )
    discipline.check_jacobian()


def _r3(r):
    return r**3


def _der_r3(x, norx, eps):
    return 3.0 * x * norx / eps**3


@pytest.mark.parametrize("transformer", TRANSFORMERS)
@pytest.mark.parametrize("function", [*list(RBF), _r3])
def test_rbf(dataset, transformer, function) -> None:
    """Test polynomial regression Jacobians."""
    der_func = _der_r3 if function is _r3 else None

    discipline = SurrogateDiscipline(
        "RBFRegressor",
        data=dataset,
        transformer=transformer,
        function=function,
        der_function=der_func,
    )
    discipline.check_jacobian()


def test_pce(dataset) -> None:
    """Test polynomial regression Jacobians."""
    space = ParameterSpace()

    for input_name in dataset.get_variable_names(dataset.INPUT_GROUP):
        space.add_random_variable(input_name, "OTUniformDistribution")

    discipline = SurrogateDiscipline(
        "PCERegressor", data=dataset, transformer={}, probability_space=space
    )
    discipline.check_jacobian()
