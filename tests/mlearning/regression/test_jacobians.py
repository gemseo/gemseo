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
#        :author: Syver Doeving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
""" Test analytical Jacobian expressions against finite difference
approximations. This is done using the built in check method of MDODiscipline.
The regression models are thus converted to surrogate disciplines. The
Jacobians are checked over different combinations of datasets (scalar and
vector inputs and outputs), transformers and parameters.
"""
from __future__ import absolute_import, division, unicode_literals

from itertools import product

import pytest
from future import standard_library
from numpy import arange

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.dataset import Dataset
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.surrogate_disc import SurrogateDiscipline
from gemseo.mlearning.regression.gpr import GaussianProcessRegression
from gemseo.mlearning.regression.rbf import RBFRegression
from gemseo.mlearning.transform.dimension_reduction.pca import PCA
from gemseo.mlearning.transform.scaler.scaler import Scaler

standard_library.install_aliases()

LEARNING_SIZE = 100


@pytest.fixture
def dataset_scalar_scalar():
    """ Dataset from a R -> R function sampled over [0,1]^2. """
    expressions_dict = {"y_1": "1+3*x_1"}
    discipline = AnalyticDiscipline("func", expressions_dict)
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=-3.0, u_b=3.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "lhs", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("scalar_scalar")


@pytest.fixture
def dataset_scalar_vector():
    """ Dataset from a R -> R^3 function sampled over [0,1]^2. """
    expressions_dict = {"y_1": "1+2*x_1", "y_2": "-1-2*x_1", "y_3": "3"}
    discipline = AnalyticDiscipline("func", expressions_dict)
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=-3.0, u_b=3.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_2", design_space)
    scenario.execute({"algo": "lhs", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("scalar_vector")


@pytest.fixture
def dataset_vector_scalar():
    """ Dataset from a R^2 -> R function sampled over [0,1]^2. """
    expressions_dict = {"y_1": "1+2*x_1+3*x_2"}
    discipline = AnalyticDiscipline("func", expressions_dict)
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=-3.0, u_b=3.0)
    design_space.add_variable("x_2", l_b=-3.0, u_b=3.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "lhs", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("vector_scalar")


@pytest.fixture
def dataset_linear():
    """ Dataset from a R^2 -> R^3 function sampled over [0,1]^2. """
    expressions_dict = {"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2", "y_3": "3"}
    discipline = AnalyticDiscipline("func", expressions_dict)
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=-3.0, u_b=3.0)
    design_space.add_variable("x_2", l_b=-3.0, u_b=3.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "lhs", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("linear")


@pytest.fixture
def dataset_polynomial():
    """ Dataset from a R^3 -> R^3 function sampled over [0,1]^2. """
    expressions_dict = {
        "y_1": "1+2*x_1+3*x_2 + 0.5*x_1**5 + 4*x_1*x_2**3",
        "y_2": "-1-2*x_1-3*x_2 - 0.5*x_2**4+ 7*x_1**3*x_3**2",
        "y_3": "3-9*x_3**2",
    }
    discipline = AnalyticDiscipline("func", expressions_dict)
    discipline.set_cache_policy(discipline.MEMORY_FULL_CACHE)
    design_space = DesignSpace()
    design_space.add_variable("x_1", l_b=-3.0, u_b=3.0)
    design_space.add_variable("x_2", l_b=-3.0, u_b=3.0)
    design_space.add_variable("x_3", l_b=-4.0, u_b=4.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y_1", design_space)
    scenario.execute({"algo": "lhs", "n_samples": LEARNING_SIZE})
    return discipline.cache.export_to_dataset("polynomial")


@pytest.fixture
def datasets(
    dataset_scalar_scalar,
    dataset_scalar_vector,
    dataset_vector_scalar,
    dataset_linear,
    dataset_polynomial,
):
    """ List of different datasets. """
    return [
        dataset_scalar_scalar,
        dataset_scalar_vector,
        dataset_vector_scalar,
        dataset_linear,
        dataset_polynomial,
    ]


@pytest.fixture
def transformers():
    """ List of different transformers. """
    return [
        {},
        {Dataset.INPUT_GROUP: Scaler(offset=5, coefficient=3)},
        {Dataset.OUTPUT_GROUP: Scaler(offset=-3, coefficient=0.1)},
        {
            Dataset.INPUT_GROUP: Scaler(offset=10, coefficient=4),
            Dataset.OUTPUT_GROUP: Scaler(offset=-7, coefficient=-8),
        },
        {
            Dataset.INPUT_GROUP: PCA(n_components=1),
            Dataset.OUTPUT_GROUP: PCA(n_components=1),
        },
    ]


def test_linreg(datasets, transformers):
    """ Test linear regression Jacobians. """
    intercepts = [True, False]
    for dataset, transformer, fit_intercept in product(
        datasets, transformers, intercepts
    ):
        discipline = SurrogateDiscipline(
            "LinearRegression",
            data=dataset,
            transformer=transformer,
            fit_intercept=fit_intercept,
        )
        discipline.check_jacobian()


def test_polyreg(datasets, transformers):
    """ Test polynomial regression Jacobians. """
    intercepts = [True, False]
    degrees = arange(1, 5)
    for dataset, transformer, fit_intercept, degree in product(
        datasets, transformers, intercepts, degrees
    ):
        discipline = SurrogateDiscipline(
            "PolynomialRegression",
            data=dataset,
            transformer=transformer,
            fit_intercept=fit_intercept,
            degree=degree,
        )
        discipline.check_jacobian()


def test_rbf(datasets, transformers):
    """ Test polynomial regression Jacobians. """
    functions = RBFRegression.AVAILABLE_FUNCTIONS + [lambda r: r ** 3]
    der_funcs = [None] * len(functions)
    der_funcs[-1] = lambda x, norx, eps: 3.0 * x * norx / eps ** 3
    funcs_and_ders = zip(functions, der_funcs)
    for dataset, transformer, (func, der_func) in product(
        datasets, transformers, funcs_and_ders
    ):
        discipline = SurrogateDiscipline(
            "RBFRegression",
            data=dataset,
            transformer=transformer,
            function=func,
            der_function=der_func,
        )
        discipline.check_jacobian()


def test_pce(datasets, transformers):
    """ Test polynomial regression Jacobians. """

    for dataset in datasets:
        space = ParameterSpace()
        for input_name in dataset.get_names(dataset.INPUT_GROUP):
            space.add_random_variable(input_name, "OTUniformDistribution")
        discipline = SurrogateDiscipline(
            "PCERegression", data=dataset, transformer=None, probability_space=space
        )
        discipline.check_jacobian()


def test_gpr(datasets, transformers):
    """ Test Gaussian process regression Jacobians (not implemented). """
    for dataset, transformer in product(datasets, transformers):
        discipline = SurrogateDiscipline(
            "GaussianProcessRegression", data=dataset, transformer=transformer
        )
        with pytest.raises(NotImplementedError):
            discipline.check_jacobian()


def test_random_forest(datasets, transformers):
    """ Test random forest regression Jacobians (not implemented) """
    for dataset, transformer in product(datasets, transformers):
        discipline = SurrogateDiscipline(
            "RandomForestRegressor", data=dataset, transformer=transformer
        )
        with pytest.raises(NotImplementedError):
            discipline.check_jacobian()
