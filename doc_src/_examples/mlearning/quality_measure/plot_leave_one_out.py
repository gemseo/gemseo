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
Leave-one-out
=============
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from numpy import array
from numpy import linspace
from numpy import newaxis
from numpy import sin

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.polyreg import PolynomialRegressor
from gemseo.mlearning.regression.quality.rmse_measure import RMSEMeasure

# %%
# Every quality measure can be computed from a training dataset or a test dataset.
# The use of a test dataset aims to
# approximate the quality of the machine learning model over the whole variable space
# in order to be less dependent on the training dataset
# and so to avoid over-fitting (accurate near learning points and poor elsewhere).
#
# In the presence of expensive data,
# this test dataset may just be a dream,
# and we have to estimate this quality with techniques resampling the training dataset,
# such as leave-one-out.
# The idea is simple:
# we repeat iterate :math:`N` times the two-step task
# "1) learn from :math:`N-1` samples, 2) predict from the remainder"
# and finally approximate the measure from the :math:`N` predictions.
#
# To illustrate this point,
# let us consider the function :math:`f(x)=(6x-2)^2\sin(12x-4)` :cite:`forrester2008`:


def f(x):
    return (6 * x - 2) ** 2 * sin(12 * x - 4)


# %%
# and try to approximate it with a polynomial of order 3.
#
# For this,
# we can take these :math:`N=7` learning input points
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
# Finally,
# we compute the quality of this model with the RMSE metric:
rmse = RMSEMeasure(polynomial)
rmse.compute_learning_measure()

# %%
# As the cost of this academic function is zero,
# we can approximate the generalization quality with a large test dataset
# whereas the usual test size is about 20% of the training size.
x_test = linspace(0.0, 1.0, 100)
y_test = f(x_test)
dataset_test = IODataset()
dataset_test.add_input_group(x_test[:, newaxis], ["x"])
dataset_test.add_output_group(y_test[:, newaxis], ["y"])
rmse.compute_test_measure(dataset_test)

# %%
# And do the same by leave-one-validation
rmse.compute_leave_one_out_measure(store_resampling_result=True)

# %%
# In this case,
# the leave-one-out error is very pessimistic.
# We can take a closer look by storing the :math:`N` sub-models:
rmse.compute_leave_one_out_measure(store_resampling_result=True)

# %%
# and plotting their outputs:
plot = plt.plot(x_test, y_test, label="Reference")
plt.plot(x_train, y_train, "o", color=plot[0].get_color(), label="Training dataset")
plt.plot(x_test, polynomial.predict(x_test[:, newaxis]), label="Model")
for i, algo in enumerate(polynomial.resampling_results["LeaveOneOut"][1], 1):
    plt.plot(x_test, algo.predict(x_test[:, newaxis]), label=f"Sub-model {i}")
plt.legend()
plt.grid()
plt.show()

# %%
# We can see that
# this pessimistic error is mainly due to the second sub-model
# which did not learn the first training point
# and therefore has a very high extrapolation error.
