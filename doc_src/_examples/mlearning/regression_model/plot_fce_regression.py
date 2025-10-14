# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
r"""
Function chaos expansion
========================

Given a training dataset
whose input samples are generated from OpenTURNS probability distributions,
the :class:`.FCERegressor` can use any linear model fitting algorithm,
including sparse techniques,
to fit a functional chaos expansion (FCE) model of the form

.. math::

   Y = \sum_{i\in\mathcal{I}\subset\mathbb{N}^d} w_i\Psi_i(X)

where :math:`\Psi_i(X)=\prod_{j=1}^d\psi_{i,j}(X_j)`
and :math:`\mathbb{E}[\Psi_i(X)\Psi_j(X)]=\delta_{ij}`
with :math:`\delta` the Kronecker delta and :math:`X` a random vector.

A particular version of FCE is the polynomial chaos expansion (PCE)
for which the class :class:`.PCERegressor` interfaces
the OpenTURNS algorithm :class:`openturns.FunctionalChaosAlgorithm`
(see the `OpenTURNS documentation <https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.FunctionalChaosAlgorithm.html?highlight=functionalchaosalgorithm#openturns.FunctionalChaosAlgorithm>`__).

Note that FCE can also learn Jacobian data
in the hope of improving the quality of the surrogate model
for the same evaluation budget.

In this example,
we will compare different types of :class:`.FCERegressor`
to approximate the Ishigami function

.. math::
   f(X) = \sin(X_1) + 7\sin(X_2)^2 + 0.1X_3^4\sin(X_1)

where :math:`X_1`, :math:`X_2` and :math:`X_3`
are independent and uniformly distributed over the interval :math:`[-\pi,\pi]`.
"""

from __future__ import annotations

from numpy import array

from gemseo import sample_disciplines
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.datasets.dataset import Dataset
from gemseo.mlearning.linear_model_fitting.elastic_net_cv_settings import (
    ElasticNetCV_Settings,
)
from gemseo.mlearning.linear_model_fitting.lars_cv_settings import LARSCV_Settings
from gemseo.mlearning.linear_model_fitting.lasso_cv_settings import LassoCV_Settings
from gemseo.mlearning.linear_model_fitting.linear_regression_settings import (
    LinearRegression_Settings,
)
from gemseo.mlearning.linear_model_fitting.null_space_settings import NullSpace_Settings
from gemseo.mlearning.linear_model_fitting.omp_cv_settings import (
    OrthogonalMatchingPursuitCV_Settings,
)
from gemseo.mlearning.linear_model_fitting.ridge_cv_settings import RidgeCV_Settings
from gemseo.mlearning.linear_model_fitting.spgl1_settings import SPGL1_Settings
from gemseo.mlearning.regression.algos.fce import FCERegressor
from gemseo.mlearning.regression.algos.fce_settings import FCERegressor_Settings
from gemseo.mlearning.regression.algos.fce_settings import OrthonormalFunctionBasis
from gemseo.mlearning.regression.algos.pce import PCERegressor
from gemseo.mlearning.regression.algos.pce_settings import PCERegressor_Settings
from gemseo.mlearning.regression.quality.r2_measure import R2Measure
from gemseo.post.dataset.bars import BarPlot
from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace

# %%
# First,
# we define the Ishigami discipline and its uncertain space:
discipline = IshigamiDiscipline()
uncertain_space = IshigamiSpace(IshigamiSpace.UniformDistribution.OPENTURNS)

# %%
# and create a training dataset using an optimized latin hypercube sampling:
training_dataset = sample_disciplines(
    [discipline],
    uncertain_space,
    "y",
    algo_settings_model=OT_OPT_LHS_Settings(n_samples=70, eval_jac=True),
)

# %%
# as well as a validation dataset using Monte Carlo sampling:
validation_dataset = sample_disciplines(
    [discipline],
    uncertain_space,
    "y",
    algo_settings_model=MC_Settings(n_samples=1000),
)

# %%
# Then,
# we create standard and gradient-enhanced FCEs
# using an orthonormal polynomial basis (default basis)
# with a maximum total degree of 7
# and different regression techniques from scikit-learn to estimate the coefficients,
# namely `ordinary least squares <https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares>`__,
# `ridge <https://scikit-learn.org/stable/modules/linear_model.html#regression>`__ (i.e., L2 regularisation),
# `lasso <https://scikit-learn.org/stable/modules/linear_model.html#lasso>`__ (i.e., L1 regularisation),
# `elasticnet <https://scikit-learn.org/stable/modules/linear_model.html#elastic-net>`__ (i.e., L1 and L2 regularisation),
# `least angle regression <https://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression>`__ (LARS)
# and `orthogonal matching pursuit <https://scikit-learn.org/stable/modules/linear_model.html#orthogonal-matching-pursuit-omp>`__.
# Note that all these algorithms have been finely tuned using cross-validation,
# except ordinary least squares regression for which there is no parameter to tune.
# We also add the `SPGL1 algorithm <https://friedlander.io/spgl1/>`__
# to solve a basis pursuit denoise (BPN) problem,
# as well as a null space algorithm :cite:`ghisu2021`.
r2_learning = []
r2_validation = []
r2_learning_ge = []
r2_validation_ge = []
null_space_settings = NullSpace_Settings()
for linear_model_fitter_settings in [
    LinearRegression_Settings(),
    RidgeCV_Settings(),
    LassoCV_Settings(),
    ElasticNetCV_Settings(),
    LARSCV_Settings(),
    OrthogonalMatchingPursuitCV_Settings(),
    SPGL1_Settings(sigma=1e-7),
    null_space_settings,
]:
    if linear_model_fitter_settings == null_space_settings:
        # The null space technique requires gradient observations.
        r2_learning.append(0.0)
        r2_validation.append(0.0)
    else:
        # Train an FCE.
        fce_settings = FCERegressor_Settings(
            degree=7,
            linear_model_fitter_settings=linear_model_fitter_settings,
        )
        fce = FCERegressor(training_dataset, fce_settings)
        fce.learn()

        # Assess the quality of the FCE.
        r2 = R2Measure(fce)
        r2_learning.append(r2.compute_learning_measure().round(2)[0])
        r2_validation.append(r2.compute_test_measure(validation_dataset).round(2)[0])

    # Train a gradient-enhanced FCE.
    fce_settings = FCERegressor_Settings(
        degree=7,
        linear_model_fitter_settings=linear_model_fitter_settings,
        learn_jacobian_data=True,
    )
    fce = FCERegressor(training_dataset, fce_settings)
    fce.learn()

    # Assess the quality of the gradient-enhanced FCE.
    r2 = R2Measure(fce)
    r2_learning_ge.append(r2.compute_learning_measure().round(2)[0])
    r2_validation_ge.append(r2.compute_test_measure(validation_dataset).round(2)[0])

# %%
# We create also a :class:`.PCERegressor`
# using the LARS algorithm implemented in OpenTURNS:
pce = PCERegressor(training_dataset, PCERegressor_Settings(degree=7, use_lars=True))
pce.learn()
r2 = R2Measure(pce)
r2_learning.append(r2.compute_learning_measure().round(2)[0])
r2_validation.append(r2.compute_test_measure(validation_dataset).round(2)[0])
r2_learning_ge.append(0)
r2_validation_ge.append(0)

# %%
# From these results,
# we can plot the quality of the different surrogate models,
# expressed in terms of coefficient of determination :math:`R^2`
# (the higher, the better):
dataset = Dataset()
dataset.add_group(
    "R2",
    array([r2_learning, r2_validation, r2_learning_ge, r2_validation_ge]),
    ("OLS", "L2", "L1", "L1-L2", "LARS", "OMP", "SPGL1", "NullSpace", "OT-LARS"),
)
dataset.index = ["Learning", "Validation", "Learning-GE", "Validation-GE"]

barplot = BarPlot(dataset, annotate=False)
barplot.execute(save=False)

# %%
# First,
# let us focus on the standard FCEs
# that have not learned derivatives ("Learning" and "Validation" in the legend).
# We can see that
# the quality of learning is perfect, regardless of the method.
# That's good, but not enough.
# But what interests us is the quality of prediction of the validation dataset
# to see if the surrogate model avoids overfitting.
# In this regard,
# ordinary least squares regression and ridge regression are wrong
# while the other techniques are very good,
# without really being able to tell them apart.
# Now,
# if we have a look to the gradient-enhanced FCEs
# ("Learning-GE" and "Validation-GE" in the legend).
# we can see that the quality is significantly better, except for the LARS method.
#
# Lastly,
# these numerical experiments can be repeated
# by replacing the polynomial basis with the Fourier series.
r2_learning = []
r2_validation = []
r2_learning_ge = []
r2_validation_ge = []
null_space_settings = NullSpace_Settings()
for linear_model_fitter_settings in [
    LinearRegression_Settings(),
    RidgeCV_Settings(),
    LassoCV_Settings(),
    ElasticNetCV_Settings(),
    LARSCV_Settings(),
    OrthogonalMatchingPursuitCV_Settings(),
    SPGL1_Settings(sigma=1e-7),
    null_space_settings,
]:
    if linear_model_fitter_settings == null_space_settings:
        # The null space technique requires gradient observations.
        r2_learning.append(0.0)
        r2_validation.append(0.0)
    else:
        # Train an FCE.
        fce_settings = FCERegressor_Settings(
            degree=7,
            linear_model_fitter_settings=linear_model_fitter_settings,
            basis=OrthonormalFunctionBasis.FOURIER,
        )
        fce = FCERegressor(training_dataset, fce_settings)
        fce.learn()

        # Assess the quality of the FCE.
        r2 = R2Measure(fce)
        r2_learning.append(r2.compute_learning_measure().round(2)[0])
        r2_validation.append(r2.compute_test_measure(validation_dataset).round(2)[0])

    # Train a gradient-enhanced FCE.
    fce_settings = FCERegressor_Settings(
        degree=7,
        linear_model_fitter_settings=linear_model_fitter_settings,
        basis=OrthonormalFunctionBasis.FOURIER,
        learn_jacobian_data=True,
    )
    fce = FCERegressor(training_dataset, fce_settings)
    fce.learn()

    # Assess the quality of the gradient-enhanced FCE.
    r2 = R2Measure(fce)
    r2_learning_ge.append(r2.compute_learning_measure().round(2)[0])
    r2_validation_ge.append(r2.compute_test_measure(validation_dataset).round(2)[0])

dataset = Dataset()
dataset.add_group(
    "R2",
    array([r2_learning, r2_validation, r2_learning_ge, r2_validation_ge]),
    ("OLS", "L2", "L1", "L1-L2", "LARS", "OMP", "SPGL1", "NullSpace"),
)
dataset.index = ["Learning", "Validation", "Learning-GE", "Validation-GE"]

barplot = BarPlot(dataset, annotate=False)
barplot.execute(save=False)

# %%
# We then see the same type of ranking, with even better validation qualities.
# This can be easily explained by the nature of Ishigami's function,
# in which trigonometric terms are important.
# Furthermore,
# learning Jacobian significantly improves the quality of surrogate models
# in the case of ridge regression and ordinary least squares.
