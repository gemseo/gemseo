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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test mean squared error measure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import allclose

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.regression.algos.polyreg import PolynomialRegressor
from gemseo.mlearning.regression.quality.mse_measure import MSEMeasure
from gemseo.mlearning.regression.quality.rmse_measure import RMSEMeasure
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.utils.testing.helpers import concretize_classes

from ..core.test_ml_algo import DummyMLAlgo

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset

MODEL = AnalyticDiscipline({"y": "1+x+x**2"})
MODEL.set_cache(MODEL.CacheType.MEMORY_FULL)

TOL_DEG_1 = 0.03
TOL_DEG_2 = 0.001
ATOL = 1e-12


@pytest.fixture
def dataset() -> Dataset:
    """The dataset used to train the regression algorithms."""
    MODEL.cache.clear()
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    scenario = DOEScenario(
        [MODEL], "y", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=20)
    return MODEL.cache.to_dataset()


@pytest.fixture
def dataset_test() -> Dataset:
    """The dataset used to test the performance of the regression algorithms."""
    MODEL.cache.clear()
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    scenario = DOEScenario(
        [MODEL], "y", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.execute(algo_name="PYDOE_FULLFACT", n_samples=5)
    return MODEL.cache.to_dataset()


def test_constructor(dataset) -> None:
    """Test construction."""
    with concretize_classes(DummyMLAlgo):
        algo = DummyMLAlgo(dataset)

    measure = MSEMeasure(algo)
    assert measure.algo is not None
    assert measure.algo.learning_set is dataset


def test_compute_learning_measure(dataset) -> None:
    """Test evaluate learn method."""
    algo = PolynomialRegressor(dataset, degree=2)
    measure = MSEMeasure(algo)
    mse_train = measure.compute_learning_measure()
    assert mse_train < TOL_DEG_2
    measure = RMSEMeasure(algo)
    rmse_train = measure.compute_learning_measure()
    assert abs(mse_train**0.5 - rmse_train) < 1e-6

    algo = PolynomialRegressor(dataset, degree=1)
    measure = MSEMeasure(algo)
    mse_train = measure.compute_learning_measure()
    assert mse_train < TOL_DEG_1

    algo = PolynomialRegressor(
        dataset,
        degree=2,
        transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()},
    )
    measure = MSEMeasure(algo)
    mse_train = measure.compute_learning_measure()
    assert mse_train < TOL_DEG_2


def test_compute_test_measure(dataset, dataset_test) -> None:
    """Test evaluate test method."""
    algo = PolynomialRegressor(dataset, degree=2)
    mse_test = MSEMeasure(algo).compute_test_measure(dataset_test)
    assert mse_test < TOL_DEG_2
    assert (
        abs(mse_test**0.5 - RMSEMeasure(algo).compute_test_measure(dataset_test)) < 1e-6
    )

    algo = PolynomialRegressor(dataset, degree=1)
    assert MSEMeasure(algo).compute_test_measure(dataset_test) < TOL_DEG_1

    algo = PolynomialRegressor(
        dataset,
        degree=2,
        transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()},
    )
    assert MSEMeasure(algo).compute_test_measure(test_data=dataset_test) < TOL_DEG_2


def test_compute_leave_one_out_measure(dataset) -> None:
    """Test evaluate leave one out method."""
    algo = PolynomialRegressor(dataset, degree=2)
    measure = MSEMeasure(algo)
    mse_loo = measure.compute_leave_one_out_measure()
    assert mse_loo < TOL_DEG_2
    measure = RMSEMeasure(algo)
    rmse_loo = measure.compute_leave_one_out_measure()
    assert abs(mse_loo**0.5 - rmse_loo) < 1e-6

    algo = PolynomialRegressor(dataset, degree=1)
    measure = MSEMeasure(algo)
    mse_loo = measure.compute_leave_one_out_measure()
    assert mse_loo < TOL_DEG_1


def test_compute_cross_validation_measure(dataset) -> None:
    """Test evaluate k-folds method."""
    algo = PolynomialRegressor(dataset, degree=2)
    measure = MSEMeasure(algo)
    mse_kfolds = measure.compute_cross_validation_measure()
    assert mse_kfolds < TOL_DEG_2
    measure = RMSEMeasure(algo)
    rmse_kfolds = measure.compute_cross_validation_measure()
    assert abs(mse_kfolds**0.5 - rmse_kfolds) < 1e-6

    algo = PolynomialRegressor(dataset, degree=1)
    measure = MSEMeasure(algo)
    mse_kfolds = measure.compute_cross_validation_measure()
    assert mse_kfolds < TOL_DEG_1

    algo = PolynomialRegressor(
        dataset,
        degree=2,
        transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()},
    )
    measure = MSEMeasure(algo, fit_transformers=True)
    mse_kfolds = measure.compute_cross_validation_measure()
    assert mse_kfolds < TOL_DEG_2


def test_compute_bootstrap_measure(dataset) -> None:
    """Test evaluate bootstrap method."""
    algo = PolynomialRegressor(dataset, degree=2)
    measure = MSEMeasure(algo)
    mse_bootstrap = measure.compute_bootstrap_measure()
    assert mse_bootstrap < TOL_DEG_2
    measure = RMSEMeasure(algo)
    rmse_bootstrap = measure.compute_bootstrap_measure()
    rmse_bootstrap_2 = measure.compute_bootstrap_measure(samples=list(range(20)))
    assert abs(rmse_bootstrap - rmse_bootstrap_2) < 1e-6
    assert abs(mse_bootstrap**0.5 - rmse_bootstrap) < 1e-6

    algo = PolynomialRegressor(dataset, degree=1)
    measure = MSEMeasure(algo)
    mse_bootstrap = measure.compute_bootstrap_measure()
    assert mse_bootstrap < TOL_DEG_1


@pytest.mark.parametrize(
    ("method_name", "class_name"),
    [
        ("compute_bootstrap_measure", "Bootstrap"),
        ("compute_cross_validation_measure", "CrossValidation"),
    ],
)
@pytest.mark.parametrize("fit", [False, True])
def test_fit_transformers(algo_for_transformer, class_name, method_name, fit) -> None:
    """Check that the user can fit the transformers with the sub-datasets.

    Otherwise, use the transformers of the assessed algorithm as they are.
    """
    mse = MSEMeasure(algo_for_transformer)
    method = getattr(mse, method_name)
    method(seed=0, store_resampling_result=True)
    model = algo_for_transformer.resampling_results[class_name][1][0]
    mse = MSEMeasure(algo_for_transformer, fit_transformers=fit)
    method = getattr(mse, method_name)
    method(seed=0, store_resampling_result=True)
    new_model = algo_for_transformer.resampling_results[class_name][1][0]
    assert (model.algo.y_train_.sum() == new_model.algo.y_train_.sum()) is not fit


@pytest.mark.parametrize(
    "method", ["compute_bootstrap_measure", "compute_cross_validation_measure"]
)
@pytest.mark.parametrize("seed", [None, 0])
def test_seed(algo_for_transformer, method, seed) -> None:
    """Check that the seed is correctly used."""
    m = MSEMeasure(algo_for_transformer)
    evaluate = getattr(m, method)
    kwargs = {"seed": seed}
    if method == "kfolds":
        kwargs["randomize"] = True

    assert allclose(evaluate(**kwargs), evaluate(**kwargs)) is (seed is not None)
