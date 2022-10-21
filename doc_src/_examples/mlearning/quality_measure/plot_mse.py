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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
MSE example - test-train split
==============================

In this example we consider a polynomial linear regression, splitting the data
into two sets. We measure the quality of the regression by comparing the
predictions with the output on the test set.

"""
from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.api import configure_logger
from gemseo.api import create_dataset
from gemseo.core.dataset import Dataset
from gemseo.mlearning.api import create_regression_model
from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure
from numpy import arange
from numpy import argmin
from numpy import hstack
from numpy import linspace
from numpy import sort
from numpy.random import choice
from numpy.random import normal
from numpy.random import seed

configure_logger()

###############################################################################
# Define parameters
# -----------------
seed(12345)
n_samples = 10
noise = 0.3**2
max_pow = 5
amount_train = 0.8

###############################################################################
# Construct data
# --------------
# We construct a parabola with added noise, on the interval [0, 1].


def f(x):
    return -4 * (x - 0.5) ** 2 + 3


x = linspace(0, 1, n_samples)
y = f(x) + normal(0, noise, n_samples)

###############################################################################
# Indices for test-train split
# ----------------------------
samples = arange(n_samples)
n_train = int(amount_train * n_samples)
n_test = n_samples - n_train
train = sort(choice(samples, n_train, False))
test = sort([sample for sample in samples if sample not in train])
print("Train:", train)
print("Test:", test)

###############################################################################
# Build datasets
# --------------
data = hstack([x[:, None], y[:, None]])
variables = ["x", "y"]
groups = {"x": Dataset.INPUT_GROUP, "y": Dataset.OUTPUT_GROUP}
dataset = create_dataset("synthetic_data", data[train], variables, groups=groups)
dataset_test = create_dataset("synthetic_data", data[test], variables, groups=groups)

###############################################################################
# Build regression model
# ----------------------
model = create_regression_model("PolynomialRegressor", dataset, degree=max_pow)
print(model)

###############################################################################
# Predictions errors
# ------------------
measure = MSEMeasure(model)

mse_train = measure.evaluate("learn")
mse_test = measure.evaluate("test", test_data=dataset_test)

print("Training error:", mse_train)
print("Test error:", mse_test)

###############################################################################
# Compute predictions
# -------------------
measure = MSEMeasure(model)
model.learn()

n_refined = 1000
x_refined = linspace(0, 1, n_refined)
y_refined = model.predict({"x": x_refined[:, None]})["y"].flatten()

###############################################################################
# Plot data points
# ----------------
plt.plot(x_refined, f(x_refined), label="Exact function")
plt.scatter(x, y, label="Data points")
plt.legend()
plt.show()

###############################################################################
# Plot predictions
# ----------------
plt.plot(x_refined, y_refined, label=f"Prediction (x^{max_pow})")
plt.scatter(x[train], y[train], label="Train")
plt.scatter(x[test], y[test], color="r", label="Test")
plt.legend()
plt.show()

###############################################################################
# Compare different parameters
# ----------------------------
powers = [1, 2, 3, 4, 5, 7]
test_errors = []
for power in powers:
    model = create_regression_model("PolynomialRegressor", dataset, degree=power)
    measure = MSEMeasure(model)

    test_mse = measure.evaluate("test", test_data=dataset_test)
    test_errors += [test_mse]

    y_refined = model.predict({"x": x_refined[:, None]})["y"].flatten()

    plt.plot(x_refined, y_refined, label=f"x^{power}")

plt.scatter(x[train], y[train], label="Train")
plt.scatter(x[test], y[test], color="r", label="Test")
plt.legend()
plt.show()

###############################################################################
# Grid search
print(test_errors)
print("Power for minimal test error:", argmin(test_errors))
