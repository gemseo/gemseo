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
"""
Machine learning algorithm selection example
============================================

In this example we use the :class:`.MLAlgoSelection` class to perform a grid
search over different algorithms and hyperparameter values.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy import linspace
from numpy import sort
from numpy.random import default_rng

from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.core.selection import MLAlgoSelection
from gemseo.mlearning.quality_measures.mse_measure import MSEMeasure

rng = default_rng(54321)

# %%
# Build dataset
# -------------
# The data are generated from the function :math:`f(x)=x^2`.
# The input data :math:`\{x_i\}_{i=1,\cdots,20}` are chosen at random
# over the interval :math:`[0,1]`.
# The output value :math:`y_i = f(x_i) + \varepsilon_i` corresponds to
# the evaluation of :math:`f` at :math:`x_i`
# corrupted by a Gaussian noise :math:`\varepsilon_i`
# with zero mean and standard deviation :math:`\sigma=0.05`.
n = 20
x = sort(rng.random(n))
y = x**2 + rng.normal(0, 0.05, n)

dataset = IODataset()
dataset.add_variable("x", x[:, None], dataset.INPUT_GROUP)
dataset.add_variable("y", y[:, None], dataset.OUTPUT_GROUP)

# %%
# Build selector
# --------------
# We consider three regression models, with different possible hyperparameters.
# A mean squared error quality measure is used with a k-folds cross validation
# scheme (5 folds).
selector = MLAlgoSelection(
    dataset, MSEMeasure, measure_evaluation_method_name="KFOLDS", n_folds=5
)
selector.add_candidate(
    "LinearRegressor",
    penalty_level=[0, 0.1, 1, 10, 20],
    l2_penalty_ratio=[0, 0.5, 1],
    fit_intercept=[True],
)
selector.add_candidate(
    "PolynomialRegressor",
    degree=[2, 3, 4, 10],
    penalty_level=[0, 0.1, 1, 10],
    l2_penalty_ratio=[1],
    fit_intercept=[True, False],
)
rbf_space = DesignSpace()
rbf_space.add_variable("epsilon", 1, "float", 0.01, 0.1, 0.05)
selector.add_candidate(
    "RBFRegressor",
    calib_space=rbf_space,
    calib_algo={"algo": "fullfact", "n_samples": 16},
    smooth=[0, 0.01, 0.1, 1, 10, 100],
)

# %%
# Select best candidate
# ---------------------
best_algo = selector.select()
best_algo

# %%
# Plot results
# ------------
# Plot the best models from each candidate algorithm
finex = linspace(0, 1, 1000)
for candidate in selector.candidates:
    algo = candidate[0]
    print(algo)
    predy = algo.predict(finex[:, None])[:, 0]
    plt.plot(finex, predy, label=algo.SHORT_ALGO_NAME)
plt.scatter(x, y, label="Training points")
plt.legend()
plt.show()
