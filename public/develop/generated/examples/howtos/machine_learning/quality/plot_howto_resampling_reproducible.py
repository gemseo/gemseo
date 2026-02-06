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
"""# Make cross-validation and bootstrap reproducible

## Problem

Cross-validation shuffles the training samples before splitting,
so that
two cross-validations with the same number of folds lead to different results.
Similarly,
bootstrap is based on a pseudo-random resampling
so that
two bootstrap validations with the same number of replicates lead to different results.
How can I make these results reproducible?

## Solution

Set the argument `seed` of the
[compute_cross_validation_measure()][gemseo.machine_learning.core.quality.base_ml_model_quality.BaseMLModelQuality.compute_cross_validation_measure]
and
[compute_bootstrap_measure()][gemseo.machine_learning.core.quality.base_ml_model_quality.BaseMLModelQuality.compute_bootstrap_measure]
methods to an integer value.

## Step-by-step guide
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from gemseo import sample_disciplines
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.machine_learning.regression.models.rbf import RBFRegressor
from gemseo.machine_learning.regression.models.rbf_settings import RBFRegressor_Settings
from gemseo.machine_learning.regression.quality.r2_measure import R2Measure
from gemseo.problems.uncertainty.wing_weight.discipline import WingWeightDiscipline
from gemseo.problems.uncertainty.wing_weight.uncertain_space import (
    WingWeightUncertainSpace,
)

# %%
# ### 1. Define the reference model
#
# In this how-to guide,
# we consider the wing weight problem
# defined in [this page][gemseo.problems.uncertainty.wing_weight].

discipline = WingWeightDiscipline()
input_space = WingWeightUncertainSpace()

# %%
# ### 2. Create the training dataset
#
# We generate $3 \times d$ training samples
# using the optimized Latin hypercube sampling strategy.

training_dataset = sample_disciplines(
    [discipline],
    input_space,
    ["Ww"],
    algo_settings_model=OT_OPT_LHS_Settings(n_samples=input_space.dimension * 3),
)

# %%
# ### 2. Create the ML model
#
# We create a regressor from this training dataset,
# taking care to normalize the data to facilitate learning.

regressor = RBFRegressor(
    training_dataset,
    settings=RBFRegressor_Settings(transformer=RBFRegressor.DEFAULT_TRANSFORMER),
)
regressor.learn()

# %%
# ### 3. Evaluate its quality by resampling
#
# #### Cross-validation
#
# We assess the quality of this regressor by cross-validation.
r2 = R2Measure(regressor)
r2.compute_cross_validation_measure()

# %%
# Evaluating the quality by cross-validation a second time leads to a new result
# because of the randomization of the samples preceding their splitting.
r2.compute_cross_validation_measure()

# %%
# Set the `seed` argument to get the same result.
r2.compute_cross_validation_measure(seed=123)

# %%
# Do it again to convince yourself.
r2.compute_cross_validation_measure(seed=123)

# %%
# The same mechanism exists for bootstrap.
#
# #### Bootstrap
#
# We assess the quality of the regressor by bootstrap.
r2 = R2Measure(regressor)
r2.compute_bootstrap_measure()

# %%
# Evaluating the quality by boostrap a second time leads to a new result
# because of the randomization of the samples preceding their splitting.
r2.compute_bootstrap_measure()

# %%
# Set the `seed` argument to get the same result.
r2.compute_bootstrap_measure(seed=123)

# %%
# Do it again to convince yourself.
r2.compute_bootstrap_measure(seed=123)

# %%
# ## Summary
#
# Cross-validation and bootstrap can be made reproducible
# by setting the attribute `seed` of
# [compute_cross_validation_measure()][gemseo.machine_learning.core.quality.base_ml_model_quality.BaseMLModelQuality.compute_cross_validation_measure]
# and
# [compute_bootstrap_measure()][gemseo.machine_learning.core.quality.base_ml_model_quality.BaseMLModelQuality.compute_bootstrap_measure]
# to the desired value.
#
# ## One step further
#
# Ignoring reproducibility may be relevant
# in order to see the sensitivity of cross-validation to the order of samples.
results = [r2.compute_cross_validation_measure() for _ in range(100)]
plt.boxplot(results)
