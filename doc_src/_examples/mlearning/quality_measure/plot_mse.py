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
MSE for regression models
=========================
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from numpy import array
from numpy import linspace
from numpy import newaxis
from numpy import sin

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.polyreg import PolynomialRegressor
from gemseo.mlearning.regression.algos.rbf import RBFRegressor
from gemseo.mlearning.regression.quality.mse_measure import MSEMeasure

# %%
# Given a dataset :math:`(x_i,y_i,\hat{y}_i)_{1\leq i \leq N}`
# where :math:`x_i` is an input point,
# :math:`y_i` is an output observation
# and :math:`\hat{y}_i=\hat{f}(x_i)` is an output prediction
# computed by a regression model :math:`\hat{f}`,
# the mean squared error (MSE) metric is written
#
# .. math::
#
#   \text{MSE} = \frac{1}{N}\sum_{i=1}^N(y_i-\hat{y}_i)^2 \geq 0.
#
# The lower, the better.
# From a quantitative point of view,
# this depends on the order of magnitude of the outputs.
# The square root of this average is often easier to interpret,
# as it is expressed in the units of the output (see :class:`.RMSEMeasure`).
#
# To illustrate this quality measure,
# let us consider the function :math:`f(x)=(6x-2)^2\sin(12x-4)` :cite:`forrester2008`:


def f(x):
    return (6 * x - 2) ** 2 * sin(12 * x - 4)


# %%
# and try to approximate it with a polynomial of order 3.
#
# For this,
# we can take these 7 learning input points
x_train = array([0.1, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95])

# %%
# and evaluate the model ``f`` over this design of experiments (DOE):
y_train = f(x_train)

# %%
# Then,
# we create an :class:`.IODataset` from these 7 learning samples:
dataset_train = IODataset()
dataset_train.add_input_group(x_train[:, newaxis], ["x"])
dataset_train.add_output_group(y_train[:, newaxis], ["y"])

# %%
# and build a :class:`.PolynomialRegressor` with ``degree=3`` from it:
polynomial = PolynomialRegressor(dataset_train, degree=3)
polynomial.learn()

# %%
# Before using it,
# we are going to measure its quality with the MSE metric:
mse = MSEMeasure(polynomial)
result = mse.compute_learning_measure()
result, result**0.5 / (y_train.max() - y_train.min())

# %%
# This result is medium (14% of the learning output range),
# and we can be expected to a poor generalization quality.
# As the cost of this academic function is zero,
# we can approximate this generalization quality with a large test dataset
# whereas the usual test size is about 20% of the training size.
x_test = linspace(0.0, 1.0, 100)
y_test = f(x_test)
dataset_test = IODataset()
dataset_test.add_input_group(x_test[:, newaxis], ["x"])
dataset_test.add_output_group(y_test[:, newaxis], ["y"])
result = mse.compute_test_measure(dataset_test)
result, result**0.5 / (y_test.max() - y_test.min())

# %%
# The quality is higher than 15% of the test output range, which is pretty mediocre.
# This can be explained by a broader generalization domain
# than that of learning, which highlights the difficulties of extrapolation:
plt.plot(x_test, y_test, "-b", label="Reference")
plt.plot(x_train, y_train, "ob")
plt.plot(x_test, polynomial.predict(x_test[:, newaxis]), "-r", label="Prediction")
plt.plot(x_train, polynomial.predict(x_train[:, newaxis]), "or")
plt.legend()
plt.grid()
plt.show()

# %%
# Using the learning domain would slightly improve the quality:
x_test = linspace(x_train.min(), x_train.max(), 100)
y_test_in_large_domain = y_test
y_test = f(x_test)
dataset_test_in_learning_domain = IODataset()
dataset_test_in_learning_domain.add_input_group(x_test[:, newaxis], ["x"])
dataset_test_in_learning_domain.add_output_group(y_test[:, newaxis], ["y"])
mse.compute_test_measure(dataset_test_in_learning_domain)
result, result**0.5 / (y_test.max() - y_test.min())

# %%
# Lastly,
# to get better results without new learning points,
# we would have to change the regression model:
rbf = RBFRegressor(dataset_train)
rbf.learn()

# %%
# The quality of this :class:`.RBFRegressor` is quite good,
# both on the learning side:
mse_rbf = MSEMeasure(rbf)
result = mse_rbf.compute_learning_measure()
result, result**0.5 / (y_train.max() - y_train.min())

# %%
# and on the validation side:
result = mse_rbf.compute_test_measure(dataset_test_in_learning_domain)
result, result**0.5 / (y_test.max() - y_test.min())

# %%
# including the larger domain:
result = mse_rbf.compute_test_measure(dataset_test)
result, result**0.5 / (y_test_in_large_domain.max() - y_test_in_large_domain.min())

# %%
# A final plot to convince us:
plt.plot(x_test, y_test, "-b", label="Reference")
plt.plot(x_train, y_train, "ob")
plt.plot(x_test, rbf.predict(x_test[:, newaxis]), "-r", label="Prediction")
plt.plot(x_train, rbf.predict(x_train[:, newaxis]), "or")
plt.legend()
plt.grid()
plt.show()
