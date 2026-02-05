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
"""# Select an ML model

## Problem

Given a list of ML models,
how to select the best one?

## Solution

Use the
[MLModelSelection][gemseo.machine_learning.core.selection.MLModelSelection] algorithm.

## Step-by-step guide
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import linspace
from numpy import sort
from numpy.random import default_rng

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.datasets.io_dataset import IODataset
from gemseo.machine_learning.core.selection import MLModelSelection
from gemseo.machine_learning.regression.models.linreg_settings import (
    LinearRegressor_Settings,
)
from gemseo.machine_learning.regression.models.polyreg_settings import (
    PolynomialRegressor_Settings,
)
from gemseo.machine_learning.regression.models.rbf_settings import RBFRegressor_Settings
from gemseo.machine_learning.regression.quality.mse_measure import MSEMeasure

rng = default_rng(54321)

# %%
# ### 1. Create the training dataset
#
# The data are generated from the function $f(x)=x^2$.
# The input data $\{x_i\}_{i=1,\cdots,20}$ are chosen at random
# over the interval $[0,1]$.
# The output value $y_i = f(x_i) + \varepsilon_i$ corresponds to
# the evaluation of $f$ at $x_i$
# corrupted by a Gaussian noise $\varepsilon_i$
# with zero mean and standard deviation $\sigma=0.05$.
n = 20
x = sort(rng.random(n))
y = x**2 + rng.normal(0, 0.05, n)

training_dataset = IODataset()
training_dataset.add_variable("x", x[:, None], training_dataset.INPUT_GROUP)
training_dataset.add_variable("y", y[:, None], training_dataset.OUTPUT_GROUP)

# %%
# ### 2. Create a selection algorithm
# Use a quality measure for selection,
# e.g. mean squared error with a cross-validation.
selector = MLModelSelection(
    training_dataset, MSEMeasure, measure_evaluation_method_name="KFOLDS"
)
# %%
# Add the regression models,
# with different possible hyperparameters (the settings must be provided as lists).
selector.add_candidate(
    LinearRegressor_Settings(fit_intercept=True),
    penalty_level=[0, 0.1, 1, 10, 20],
    l2_penalty_ratio=[0, 0.5, 1],
)
selector.add_candidate(
    PolynomialRegressor_Settings(l2_penalty_ratio=1.0),
    degree=[2, 3, 4, 10],
    penalty_level=[0, 0.1, 1, 10],
    fit_intercept=[True, False],
)

# %%
# Possibly add a calibration algorithm for tuning an hyperparameter.
rbf_space = DesignSpace()
rbf_space.add_variable("epsilon", 1, "float", 0.01, 0.1, 0.05)
selector.add_candidate(
    RBFRegressor_Settings(),
    calibration_space=rbf_space,
    calibration_settings=PYDOE_FULLFACT_Settings(n_samples=16),
    smooth=[0, 0.01, 0.1, 1, 10, 100],
)

# %%
# Select the best candidate
best_model = selector.select()
best_model

# %%
# ### 3. Visualize the results
finex = linspace(0, 1, 1000)
for candidate in selector.candidates:
    model = candidate[0]
    predy = model.predict(finex[:, None])[:, 0]
    plt.plot(finex, predy, label=model.SHORT_NAME)
plt.scatter(x, y, label="Training points")
plt.legend()

# %%
# ## Summary
#
# An ML model can be selected from a list of candidates using the
# [MLModelSelection][gemseo.machine_learning.core.selection.MLModelSelection] algorithm.
#
# ## One step further
#
# At Step 3, we used ML calibration when adding the RBF candidate.
# If you want to get more inforation about this topic,
# you can read [this how-to guide][calibrate-an-ml-model].
