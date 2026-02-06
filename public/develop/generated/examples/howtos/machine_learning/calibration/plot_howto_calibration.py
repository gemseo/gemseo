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

# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Calibrate an ML model

## Problem

My ML model is not good enough.
How can I calibrate its hyperparameters to improve its quality?

## Solution

Use the
[MLModelCalibration][gemseo.machine_learning.core.calibration.MLModelCalibration] algorithm.

## Step-by-step guide
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.machine_learning.core.calibration import MLModelCalibration
from gemseo.machine_learning.regression.models.polyreg_settings import (
    PolynomialRegressor_Settings,
)
from gemseo.machine_learning.regression.quality.mse_measure import MSEMeasure
from gemseo.problems.dataset.rosenbrock import create_rosenbrock_dataset

# %%
# ### 1. Create the training dataset
training_dataset = create_rosenbrock_dataset(opt_naming=False, n_samples=25)

# %%
# ### 2. Create the test training
#
test_dataset = create_rosenbrock_dataset(opt_naming=False)

# %%
# ### 3. Calibrate the degree of a polynomial regressor
#
# #### Create the calibration space
calibration_space = DesignSpace()
calibration_space.add_variable("degree", 1, "integer", 1, 10, 1)

# %%
# #### Instantiate the calibration algorithm
calibration = MLModelCalibration(
    PolynomialRegressor_Settings(),
    training_dataset,
    calibration_space,
    MSEMeasure,
    measure_evaluation_method_name="TEST",
    measure_options={"test_data": test_dataset},
)

# %%
# #### Execute the calibration algorithm
calibration.execute(PYDOE_FULLFACT_Settings(n_samples=10))

# %%
# #### Get the main calibration results
x_opt = calibration.optimal_parameters
f_opt = calibration.optimal_criterion
degree = x_opt["degree"][0]
f"optimal degree = {degree}; optimal criterion = {f_opt}"

# %%
# #### Get the calibration history
calibration.dataset

# %%
# #### Visualize the calibration history
degree = calibration.get_history("degree")
criterion = calibration.get_history("criterion")
learning = calibration.get_history("learning")

plt.plot(degree, criterion, "-o", label="test", color="red")
plt.plot(degree, learning, "-o", label="learning", color="blue")
plt.xlabel("polynomial degree")
plt.ylabel("quality")
plt.axvline(x_opt["degree"], color="red", ls="--")
plt.legend()

# %%
# ## Summary
#
# An ML model can be calibrated using the
# [MLModelCalibration][gemseo.machine_learning.core.calibration.MLModelCalibration] algorithm,
# fed by an ML model name,
# a training dataset,
# a quality measure and a driver.
