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
"""# Disable sample shuffling prior to cross-validation

## Problem

Cross-validation shuffles the training samples before splitting,
in order to avoid problems with ordered data.
How can I use the training samples without shuffling?

## Solution

Set the argument `randomize` of the
[compute_cross_validation_measure()][gemseo.machine_learning.core.quality.base_ml_model_quality.BaseMLModelQuality.compute_cross_validation_measure]
method to `False`.

## Step-by-step guide
"""

from __future__ import annotations

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
# using optimized Latin hypercube sampling strategy.

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
r2 = R2Measure(regressor)
r2.compute_cross_validation_measure()

# %%
# Evaluating the quality by cross-validation a second time leads to a new result
# because of the randomization of the samples preceding their splitting.
r2.compute_cross_validation_measure()

# %%
# Set the `randomize` argument to `False` so as not to shuffle the samples.
r2.compute_cross_validation_measure(randomize=False)


# %%
# Let's do it again.
r2.compute_cross_validation_measure(randomize=False)

# %%
# ## Summary
#
# Cross-validation can avoid shuffling learning samples
# by setting the attribute `randomize` of
# [compute_cross_validation_measure()][gemseo.machine_learning.core.quality.base_ml_model_quality.BaseMLModelQuality.compute_cross_validation_measure]
# to `False`.
