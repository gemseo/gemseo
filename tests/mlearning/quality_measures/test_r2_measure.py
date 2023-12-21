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
"""Test R2 error measure module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import allclose

from gemseo.algos.design_space import DesignSpace
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.quality_measures.r2_measure import R2Measure
from gemseo.mlearning.regression.polyreg import PolynomialRegressor
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.utils.testing.helpers import concretize_classes

if TYPE_CHECKING:
    from gemseo.datasets.io_dataset import IODataset

MODEL = AnalyticDiscipline({"y": "1+x+x**2"})
MODEL.set_cache_policy(MODEL.CacheType.MEMORY_FULL)

TOL_DEG_1 = 0.1
TOL_DEG_2 = 0.001
TOL_DEG_3 = 0.01
ATOL = 1e-12


@pytest.fixture()
def dataset() -> IODataset:
    """The dataset used to train the regression algorithms."""
    MODEL.cache.clear()
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([MODEL], "DisciplinaryOpt", "y", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": 20})
    return MODEL.cache.to_dataset()


@pytest.fixture()
def dataset_test() -> IODataset:
    """The dataset used to test the performance of the regression algorithms."""
    MODEL.cache.clear()
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([MODEL], "DisciplinaryOpt", "y", design_space)
    scenario.execute({"algo": "fullfact", "n_samples": 5})
    return MODEL.cache.to_dataset()


def test_constructor(dataset):
    """Test construction."""
    with concretize_classes(MLAlgo):
        algo = MLAlgo(dataset)

    measure = R2Measure(algo)
    assert measure.algo is not None
    assert measure.algo.learning_set is dataset


def test_compute_learning_measure(dataset):
    """Test evaluate learn method."""
    algo = PolynomialRegressor(dataset, degree=2)
    measure = R2Measure(algo)
    r2_train = measure.compute_learning_measure()
    assert r2_train > 1 - TOL_DEG_2

    algo = PolynomialRegressor(dataset, degree=1)
    measure = R2Measure(algo)
    r2_train = measure.compute_learning_measure()
    assert r2_train > 1 - TOL_DEG_1

    algo = PolynomialRegressor(
        dataset,
        degree=2,
        transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()},
    )
    measure = R2Measure(algo)
    r2_train = measure.compute_learning_measure()
    assert r2_train > 1 - TOL_DEG_2


def test_compute_test_measure(dataset, dataset_test):
    """Test evaluate test method."""
    algo = PolynomialRegressor(dataset, degree=2)
    measure = R2Measure(algo)
    r2_test = measure.compute_test_measure(dataset_test)
    assert r2_test > 1 - TOL_DEG_2

    algo = PolynomialRegressor(dataset, degree=1)
    measure = R2Measure(algo)
    r2_test = measure.compute_test_measure(dataset_test)
    assert r2_test > 1 - TOL_DEG_1

    algo = PolynomialRegressor(
        dataset,
        degree=2,
        transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()},
    )
    measure = R2Measure(algo)
    r2_test = measure.compute_test_measure(dataset_test)
    assert r2_test > 1 - TOL_DEG_2


def test_compute_leave_one_out_measure(dataset):
    """Test evaluate leave one out method."""
    algo = PolynomialRegressor(dataset, degree=2)
    measure = R2Measure(algo)
    r2_loo = measure.compute_leave_one_out_measure()
    assert r2_loo > 1 - TOL_DEG_2

    algo = PolynomialRegressor(dataset, degree=1)
    measure = R2Measure(algo)
    r2_loo = measure.compute_leave_one_out_measure()
    assert r2_loo < 1 - TOL_DEG_3


def test_compute_cross_validation_measure(dataset):
    """Test evaluate k-folds method."""
    algo = PolynomialRegressor(dataset, degree=2)
    measure = R2Measure(algo)
    r2_kfolds = measure.compute_cross_validation_measure()
    assert r2_kfolds > 1 - TOL_DEG_2

    algo = PolynomialRegressor(dataset, degree=1)
    measure = R2Measure(algo)
    r2_kfolds = measure.compute_cross_validation_measure()
    assert r2_kfolds < 1 - TOL_DEG_3

    algo = PolynomialRegressor(
        dataset,
        degree=2,
        transformer={"inputs": MinMaxScaler(), "outputs": MinMaxScaler()},
    )
    measure = R2Measure(algo)
    r2_kfolds = measure.compute_cross_validation_measure()
    assert r2_kfolds > 1 - TOL_DEG_2


def test_compute_bootstrap_measure(dataset):
    """Test evaluate bootstrap method."""
    r2_measure = R2Measure(PolynomialRegressor(dataset, degree=2))
    assert r2_measure.compute_bootstrap_measure() == pytest.approx(1.0)


@pytest.mark.parametrize("fit", [False, True])
def test_fit_transformers(algo_for_transformer, fit):
    """Check that the transformers are fitted with the sub-datasets.

    By default, the transformers are fitted with the sub-datasets. If False, use the
    transformers of the assessed algorithm as they are.
    """
    m1 = R2Measure(algo_for_transformer)
    m2 = R2Measure(algo_for_transformer, fit_transformers=fit)
    assert (
        allclose(
            m1.compute_cross_validation_measure(seed=0),
            m2.compute_cross_validation_measure(seed=0),
        )
        is fit
    )
