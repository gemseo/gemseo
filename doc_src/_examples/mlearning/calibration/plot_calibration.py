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
"""
Calibration of a polynomial regression
======================================
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from gemseo.algos.design_space import DesignSpace
from gemseo.api import configure_logger
from gemseo.mlearning.core.calibration import MLAlgoCalibration
from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure
from gemseo.problems.dataset.rosenbrock import RosenbrockDataset
from matplotlib.tri import Triangulation

###############################################################################
# Load the dataset
# ----------------
dataset = RosenbrockDataset(opt_naming=False, n_samples=25)

###############################################################################
# Define the measure
# ------------------
configure_logger()
test_dataset = RosenbrockDataset(opt_naming=False)
measure_options = {"method": "test", "test_data": test_dataset}

###############################################################################
# Calibrate the degree of the polynomial regression
# -------------------------------------------------
# Define and execute the calibration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
calibration_space = DesignSpace()
calibration_space.add_variable("degree", 1, "integer", 1, 10, 1)
calibration = MLAlgoCalibration(
    "PolynomialRegressor",
    dataset,
    ["degree"],
    calibration_space,
    MSEMeasure,
    measure_options,
)
calibration.execute({"algo": "fullfact", "n_samples": 10})
x_opt = calibration.optimal_parameters
f_opt = calibration.optimal_criterion
print("optimal degree:", x_opt["degree"][0])
print("optimal criterion:", f_opt)

###############################################################################
# Get the history
# ^^^^^^^^^^^^^^^
print(calibration.dataset.export_to_dataframe())

###############################################################################
# Visualize the results
# ^^^^^^^^^^^^^^^^^^^^^
degree = calibration.get_history("degree")
criterion = calibration.get_history("criterion")
learning = calibration.get_history("learning")

plt.plot(degree, criterion, "-o", label="test", color="red")
plt.plot(degree, learning, "-o", label="learning", color="blue")
plt.xlabel("polynomial degree")
plt.ylabel("quality")
plt.axvline(x_opt["degree"], color="red", ls="--")
plt.legend()
plt.show()

###############################################################################
# Calibrate the ridge penalty of the polynomial regression
# --------------------------------------------------------
# Define and execute the calibration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
calibration_space = DesignSpace()
calibration_space.add_variable("penalty_level", 1, "float", 0.0, 100.0, 0.0)
calibration = MLAlgoCalibration(
    "PolynomialRegressor",
    dataset,
    ["penalty_level"],
    calibration_space,
    MSEMeasure,
    measure_options,
    degree=10,
)
calibration.execute({"algo": "fullfact", "n_samples": 10})
x_opt = calibration.optimal_parameters
f_opt = calibration.optimal_criterion
print("optimal penalty_level:", x_opt["penalty_level"][0])
print("optimal criterion:", f_opt)

###############################################################################
# Get the history
# ^^^^^^^^^^^^^^^
print(calibration.dataset.export_to_dataframe())

###############################################################################
# Visualize the results
# ^^^^^^^^^^^^^^^^^^^^^^
penalty_level = calibration.get_history("penalty_level")
criterion = calibration.get_history("criterion")
learning = calibration.get_history("learning")

plt.plot(penalty_level, criterion, "-o", label="test", color="red")
plt.plot(penalty_level, learning, "-o", label="learning", color="blue")
plt.axvline(x_opt["penalty_level"], color="red", ls="--")
plt.xlabel("ridge penalty")
plt.ylabel("quality")
plt.legend()
plt.show()

###############################################################################
# Calibrate the lasso penalty of the polynomial regression
# --------------------------------------------------------
# Define and execute the calibration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
calibration_space = DesignSpace()
calibration_space.add_variable("penalty_level", 1, "float", 0.0, 100.0, 0.0)
calibration = MLAlgoCalibration(
    "PolynomialRegressor",
    dataset,
    ["penalty_level"],
    calibration_space,
    MSEMeasure,
    measure_options,
    degree=10,
    l2_penalty_ratio=0.0,
)
calibration.execute({"algo": "fullfact", "n_samples": 10})
x_opt = calibration.optimal_parameters
f_opt = calibration.optimal_criterion
print("optimal penalty_level:", x_opt["penalty_level"][0])
print("optimal criterion:", f_opt)

###############################################################################
# Get the history
# ^^^^^^^^^^^^^^^
print(calibration.dataset.export_to_dataframe())

###############################################################################
# Visualize the results
# ^^^^^^^^^^^^^^^^^^^^^^
penalty_level = calibration.get_history("penalty_level")
criterion = calibration.get_history("criterion")
learning = calibration.get_history("learning")

plt.plot(penalty_level, criterion, "-o", label="test", color="red")
plt.plot(penalty_level, learning, "-o", label="learning", color="blue")
plt.axvline(x_opt["penalty_level"], color="red", ls="--")
plt.xlabel("lasso penalty")
plt.ylabel("quality")
plt.legend()
plt.show()

###############################################################################
# Calibrate the elasticnet penalty of the polynomial regression
# -------------------------------------------------------------
# Define and execute the calibration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
calibration_space = DesignSpace()
calibration_space.add_variable("penalty_level", 1, "float", 0.0, 40.0, 0.0)
calibration_space.add_variable("l2_penalty_ratio", 1, "float", 0.0, 1.0, 0.5)
calibration = MLAlgoCalibration(
    "PolynomialRegressor",
    dataset,
    ["penalty_level", "l2_penalty_ratio"],
    calibration_space,
    MSEMeasure,
    measure_options,
    degree=10,
)
calibration.execute({"algo": "fullfact", "n_samples": 100})
x_opt = calibration.optimal_parameters
f_opt = calibration.optimal_criterion
print("optimal penalty_level:", x_opt["penalty_level"][0])
print("optimal l2_penalty_ratio:", x_opt["l2_penalty_ratio"][0])
print("optimal criterion:", f_opt)

###############################################################################
# Get the history
# ^^^^^^^^^^^^^^^
print(calibration.dataset.export_to_dataframe())

###############################################################################
# Visualize the results
# ^^^^^^^^^^^^^^^^^^^^^
penalty_level = calibration.get_history("penalty_level").flatten()
l2_penalty_ratio = calibration.get_history("l2_penalty_ratio").flatten()
criterion = calibration.get_history("criterion").flatten()
learning = calibration.get_history("learning").flatten()

triang = Triangulation(penalty_level, l2_penalty_ratio)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.tricontourf(triang, criterion, cmap="Purples")
ax.scatter(x_opt["penalty_level"][0], x_opt["l2_penalty_ratio"][0])
ax.set_xlabel("penalty level")
ax.set_ylabel("l2 penalty ratio")
ax.set_title("Test measure")
ax = fig.add_subplot(1, 2, 2)
ax.tricontourf(triang, learning, cmap="Purples")
ax.scatter(x_opt["penalty_level"][0], x_opt["l2_penalty_ratio"][0])
ax.set_xlabel("penalty level")
ax.set_ylabel("l2 penalty ratio")
ax.set_title("Learning measure")

plt.show()

###############################################################################
# Add an optimization stage
# ^^^^^^^^^^^^^^^^^^^^^^^^^
calibration_space = DesignSpace()
calibration_space.add_variable("penalty_level", 1, "float", 0.0, 40.0, 0.0)
calibration_space.add_variable("l2_penalty_ratio", 1, "float", 0.0, 1.0, 0.5)
calibration = MLAlgoCalibration(
    "PolynomialRegressor",
    dataset,
    ["penalty_level", "l2_penalty_ratio"],
    calibration_space,
    MSEMeasure,
    measure_options,
    degree=10,
)
calibration.execute({"algo": "NLOPT_COBYLA", "max_iter": 100})
x_opt2 = calibration.optimal_parameters
f_opt2 = calibration.optimal_criterion

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.tricontourf(triang, criterion, cmap="Purples")
ax.scatter(x_opt["penalty_level"][0], x_opt["l2_penalty_ratio"][0])
ax.scatter(x_opt2["penalty_level"][0], x_opt2["l2_penalty_ratio"][0], color="red")
ax.set_xlabel("penalty level")
ax.set_ylabel("l2 penalty ratio")
ax.set_title("Test measure")
ax = fig.add_subplot(1, 2, 2)
ax.tricontourf(triang, learning, cmap="Purples")
ax.scatter(x_opt["penalty_level"][0], x_opt["l2_penalty_ratio"][0])
ax.scatter(x_opt2["penalty_level"][0], x_opt2["l2_penalty_ratio"][0], color="red")
ax.set_xlabel("penalty level")
ax.set_ylabel("l2 penalty ratio")
ax.set_title("Learning measure")
plt.show()

n_iterations = len(calibration.scenario.disciplines[0].cache)
print(f"MSE with DOE: {f_opt} (100 evaluations)")
print(f"MSE with OPT: {f_opt2} ({n_iterations} evaluations)")
print(f"MSE reduction:{round((f_opt2 - f_opt) / f_opt * 100)}%")
