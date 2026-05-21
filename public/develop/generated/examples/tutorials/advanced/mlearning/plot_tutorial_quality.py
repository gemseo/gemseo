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
r"""# Assessing the quality of an ML model

## Goal

In this tutorial,
you will learn **why and how to assess the quality of a machine learning model**.

We illustrate this tutorial using:

* an aeronautical example,
* a radial basis function (RBF) regressor,
* and quality assessment via the R² metric.

"""

from __future__ import annotations

from gemseo import sample_disciplines
from gemseo.machine_learning.regression.models.rbf import RBFRegressor
from gemseo.machine_learning.regression.quality.r2_measure import R2Measure
from gemseo.problems.uncertainty.wing_weight.discipline import WingWeightDiscipline
from gemseo.problems.uncertainty.wing_weight.uncertain_space import (
    WingWeightUncertainSpace,
)
from gemseo.settings.doe import OT_MONTE_CARLO_Settings
from gemseo.settings.doe import OT_OPT_LHS_Settings
from gemseo.settings.machine_learning import RBFRegressor_Settings

# %%
# ## Step 1 — Define the reference model
#
# In this tutorial,
# we consider the wing weight problem
# defined in [this page][gemseo.problems.uncertainty.wing_weight].
# This model computes the weight of an aircraft wing from ten inputs,
# such as the wing area and the paint weight.
# We begin by instantiating the discipline and the input space,
# the latter being an uncertain space comprising independent uniform variables.
discipline = WingWeightDiscipline()
input_space = WingWeightUncertainSpace()

# %%
# ## Step 2 — Build a small training dataset
#
# We deliberately use only $3 \times d$ training points,
# where $d=10$ is the input dimension,
# because of the supposedly high cost of $f$.
# Note that $2 \times d$ or $3 \times d$ are classic rules of thumb.
# These samples rely on an optimized Latin hypercube sampling strategy.

training_dataset = sample_disciplines(
    [discipline],
    input_space,
    ["Ww"],
    algo_settings_model=OT_OPT_LHS_Settings(n_samples=input_space.dimension * 3),
)

# %%
# ## Step 3 — Train a regression model
#
# We fit an RBF regressor using the full training dataset.
# We take care to normalize the data
# because the inputs have very different orders of magnitude,
# which could lead to a poor predictive ML model.

regressor = RBFRegressor(
    training_dataset,
    settings=RBFRegressor_Settings(transformer=RBFRegressor.DEFAULT_TRANSFORMER),
)
regressor.learn()

# %%
# ## Step 4 — Evaluate the model quality
#
# ### Learning quality
#
# We measure the prediction accuracy of this regressor
# using the coefficient of determination R²
# (the higher, the better; upper-bounded by 1)
# evaluated using the training dataset.

r2 = R2Measure(regressor)
r2.compute_learning_measure()

# %%
# This value reflects how well the model fits the training data
# but not how well it generalizes.
# We can see that the R² score is equal to 1
# because RBF regressors are interpolating by design.
# From a learning perspective,
# the model is undoubtedly excellent.
# Now,
# we need to address the error of generalization
# to avoid overfitting.
#
# ### Generalization quality
#
# Because the reference discipline is cheap to evaluate,
# we can ideally approximate the *true* generalization error
# using an independent test dataset.

test_dataset = sample_disciplines(
    [discipline],
    input_space,
    ["Ww"],
    algo_settings_model=OT_MONTE_CARLO_Settings(n_samples=1000),
)

# %%
# We compute the test R².

r2.compute_test_measure(test_dataset)

# %%
# This time,
# we can see that the quality of the model could be improved,
# even though the score displayed already indicates a very good quality model.
#
# ### Cross-validation
#
# When data is scarce or expensive to generate,
# this is not always possible.
#
# $K$-folds cross-validation (CV) validation addresses this problem by:
#
# 1. Splitting the training dataset into $K$ folds.
# 2. Removing one fold from the training set,
# 3. Training the model on the remaining folds,
# 4. Predicting the left-out fold,
# 5. Repeating the process 2-3-4 for all folds.
#
# The final error estimate aggregates the prediction errors over all left-out folds.
#
# By default,
# CV randomizes the learning samples
# before splitting them into $K=5$ folds.

r2.compute_cross_validation_measure()

# %%
# The CV R² is pessimistic, i.e. lower than the test R².
#
# ### Leave-one-out
#
# When CV use as many folds as samples,
# i.e. one sample per fold,
# we refer to as leave-one-out validation (LOO).
#
# 1. Splitting the training dataset into $n$ folds,
#    where $n$ is the number of training samples.
# 2. Removing one sample from the training set,
# 3. Training the model on the remaining samples,
# 4. Predicting the left-out sample,
# 5. Repeating the process 2-3-4 for all samples.
#
# The final error estimate aggregates the prediction errors over all left-out samples.

r2.compute_leave_one_out_measure()

# %%
#
# We can see that the LOO R² is less pessimistic than the CV one.
#
# ### Bootstrap
#
# We can also approximate the generalization error by bootstrap.
# This resampling technique creates many new training datasets
# by repeatedly sampling with replacement from the original training dataset,
# each the same size as the original.
#
# 1. Sampling with replacement $n$ samples from the original training dataset
#    where $n$ is the number of training samples.
# 2. Training the model on these samples,
# 3. Predicting the left-out samples,
# 4. Repeating $B-1$ times the process 1-2-3.
#
# The final error estimate aggregates the $B$ prediction errors.
#
# By default,
# bootstrap uses $B=100$ repetitions.
r2.compute_bootstrap_measure()

# %%
# This can see that this bootstrap R² is also pessimistic.
#
# ## Key takeaways
#
# * You must always start by calculating the learning error.
# * Estimating the generalization error is important to avoid overfitting.
# * Resampling techniques are valuable when test data is unavailable,
#   e.g. cross-validation, leave-one-out and bootstrap techniques.
# * It can overestimate the true error,
#   especially with sparse or poorly distributed samples.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [how to get the resampling result][get-the-resampling-result],
# - [how to change the number of cross-validation folds][change-the-number-of-cross-validation-folds],
# - [how to disable randomization][disable-sample-shuffling-prior-to-cross-validation],
# - [how to use deterministic randomization][make-cross-validation-and-bootstrap-reproducible].
#
