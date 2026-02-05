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

"""# Scale data before training an ML model

## Problem

Scaling data around zero is important to avoid numerical issues
when fitting a machine learning model.
This is all the more true as
the variables have different ranges
or the fitting relies on numerical optimization techniques.

How can I use a scaling policy?

## Solution

Define a scaling data transformation policy in the regressor settings.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import sample_disciplines
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.machine_learning.regression.models.gpr import GaussianProcessRegressor
from gemseo.machine_learning.regression.models.gpr_settings import (
    GaussianProcessRegressor_Settings,
)
from gemseo.machine_learning.regression.quality.r2_measure import R2Measure
from gemseo.problems.uncertainty.wing_weight.discipline import WingWeightDiscipline
from gemseo.problems.uncertainty.wing_weight.uncertain_space import (
    WingWeightUncertainSpace,
)

# %%
# ### 1. Define the reference model
#
# We consider the wing weight problem
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
# ### 3. Create the validation dataset
#
# We generate many validation samples using Monte Carlo sampling.

test_dataset = sample_disciplines(
    [discipline],
    input_space,
    ["Ww"],
    algo_settings_model=MC_Settings(n_samples=1_000),
)

# %%
# ### 4. Use an ML model without normalization
#
# Create a regressor, e.g. Gaussian process regressor (GPR).
regressor = GaussianProcessRegressor(training_dataset)
regressor.learn()

# %%
# Then, assess its quality:
r2 = R2Measure(regressor)
r2.compute_test_measure(test_dataset)

# %%
# ### 5. Use an ML model with normalization
#
# #### Scale using min-max
#
# Create a GPR from the same training dataset,
# using the default data transformation strategy,
# namely scaling both input and output variables between 0 and 1
# using the minimum and maximum values from the training dataset
# (see [MinMaxScaler][gemseo.machine_learning.transformers.scaler.min_max_scaler.MinMaxScaler]).
# Use the `DEFAULT_TRANSFORMER` attribute of the regressor for that purpose.
regressor = GaussianProcessRegressor(
    training_dataset,
    settings=GaussianProcessRegressor_Settings(
        transformer=GaussianProcessRegressor.DEFAULT_TRANSFORMER
    ),
)
regressor.learn()

# %%
# Then, assess its quality:
r2 = R2Measure(regressor)
r2.compute_test_measure(test_dataset)

# %%
# We can see that the scaling improves the R2 quality (recall: the higher, the better):
#
# #### Scale the outputs only
#
# We can also scale the outputs only:
regressor = GaussianProcessRegressor(
    training_dataset,
    settings=GaussianProcessRegressor_Settings(transformer={"outputs": "MinMaxScaler"}),
)
regressor.learn()
r2 = R2Measure(regressor)
r2.compute_test_measure(test_dataset)

# %%
# #### Scale using mean/std
#
# We can also scale the data
# by subtracting the mean and dividing by the standard deviation
# (see [StandardScaler][gemseo.machine_learning.transformers.scaler.standard_scaler.StandardScaler]).
regressor = GaussianProcessRegressor(
    training_dataset,
    settings=GaussianProcessRegressor_Settings(
        transformer={"outputs": "StandardScaler"}
    ),
)
regressor.learn()
r2 = R2Measure(regressor)
r2.compute_test_measure(test_dataset)

# %%
#
# !!! note
#
#     You can also scale a single variable
#     using the key `"y"` instead of `"outputs"`.
#
# ## Summary
#
# The ML model can be trained from scaled data,
# using scaling as data transformation policy.
# The data transformation policy can be set
# using the `transformer` parameter of the ML model settings.
# The `DEFAULT_TRANSFORMER` attribute of regressors defines a min-max scaling policy
# for both input and output variables.
#
# !!! note
#
#     GEMSEO manages the full normalization-unnormalization process:
#     - no need to provide normalized training data;
#       GEMSEO will normalize the training data using a scaling policy;
#     - do not provide normalized data to the `predict()` methods
#       under any circumstances;
#       GEMSEO will normalize the input data and unnormalize the predictions.
