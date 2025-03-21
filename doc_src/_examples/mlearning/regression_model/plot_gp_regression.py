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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Gaussian process (GP) regression
================================

A :class:`.GaussianProcessRegressor` is a GP regression model
based on `scikit-learn <https://scikit-learn.org/>`__.

.. seealso::
   You can find more information about building GP models with scikit-learn on
   `this page <https://scikit-learn.org/stable/modules/gaussian_process.html>`__.
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from numpy import array
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import sample_disciplines
from gemseo.mlearning import create_regression_model

configure_logger()

# %%
# Problem
# -------
# In this example,
# we represent the function :math:`f(x)=(6x-2)^2\sin(12x-4)` :cite:`forrester2008`
# by the :class:`.AnalyticDiscipline`
discipline = create_discipline(
    "AnalyticDiscipline",
    name="f",
    expressions={"y": "(6*x-2)**2*sin(12*x-4)"},
)
# %%
# and seek to approximate it over the input space
input_space = create_design_space()
input_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)

# %%
# To do this,
# we create a training dataset with 6 equispaced points:
training_dataset = sample_disciplines(
    [discipline], input_space, "y", algo_name="PYDOE_FULLFACT", n_samples=6
)

# %%
# Basics
# ------
# Training
# ~~~~~~~~
# Then,
# we train a GP regression model from these samples:
model = create_regression_model("GaussianProcessRegressor", training_dataset)
model.learn()

# %%
# Prediction
# ~~~~~~~~~~
# Once it is built,
# we can predict the output value of :math:`f` at a new input point:
input_value = {"x": array([0.65])}
output_value = model.predict(input_value)
output_value

# %%
# but cannot predict its Jacobian value:
try:
    model.predict_jacobian(input_value)
except NotImplementedError:
    print("The derivatives are not available for GaussianProcessRegressor.")

# %%
# Uncertainty
# ~~~~~~~~~~~
# GP models are often valued for their ability to provide model uncertainty.
# Indeed,
# a GP model is a random process fully characterized
# by its mean function
# and a covariance structure.
# Given an input point :math:`x`,
# the prediction is equal to the mean at :math:`x`
# and the uncertainty is equal to the standard deviation at :math:`x`:
standard_deviation = model.predict_std(input_value)
standard_deviation

# %%
# Plotting
# ~~~~~~~~
# You can see that the GP model interpolates the training points
# but is very bad elsewhere.
# This case-dependent problem is due to poor auto-tuning of these length scales.
# We will look at how to correct this next.
test_dataset = sample_disciplines(
    [discipline], input_space, "y", algo_name="PYDOE_FULLFACT", n_samples=100
)
input_data = test_dataset.get_view(variable_names=model.input_names).to_numpy()
reference_output_data = test_dataset.get_view(variable_names="y").to_numpy().ravel()
predicted_output_data = model.predict(input_data).ravel()
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.grid()
plt.legend()
plt.show()

# %%
# Settings
# --------
# The :class:`.GaussianProcessRegressor` has many options
# defined in the :class:`.GaussianProcessRegressor_Settings` Pydantic model.
# Here are the main ones.
#
# Kernel
# ~~~~~~
# The ``kernel`` option defines the kernel function
# parametrizing the Gaussian process regressor
# and must be passed as a scikit-learn object.
# The default kernel is the Matérn 5/2 covariance function
# with input length scales belonging to the interval :math:`[0.01,100]`,
# initialized at 1
# and optimized by the L-BFGS-B algorithm.
# We can replace this kernel by the Matérn 5/2 kernel
# with input length scales fixed at 1:
model = create_regression_model(
    "GaussianProcessRegressor",
    training_dataset,
    kernel=Matern(length_scale=1.0, length_scale_bounds="fixed", nu=2.5),
)
model.learn()
predicted_output_data_1 = model.predict(input_data).ravel()
# %%
# or a squared exponential covariance kernel
# with input length scales fixed at 1:
model = create_regression_model(
    "GaussianProcessRegressor",
    training_dataset,
    kernel=RBF(length_scale=1.0, length_scale_bounds="fixed"),
)
model.learn()
predicted_output_data_2 = model.predict(input_data).ravel()
# %%
# These two models are much better than the previous one,
# notably the one with the Matérn 5/2 kernel,
# which highlights that the concern with the initial model is
# the value of the length scales found by numerical optimization:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(
    input_data.ravel(), predicted_output_data_1, label="Regression - Kernel(Matern 2.5)"
)
plt.plot(input_data.ravel(), predicted_output_data_2, label="Regression - Kernel(RBF)")
plt.grid()
plt.legend()
plt.show()
# %%
# Bounds
# ~~~~~~
# The ``bounds`` option defines the bounds of the input length scales;
model = create_regression_model(
    "GaussianProcessRegressor", training_dataset, bounds=(1e-1, 1e2)
)
model.learn()
# %%
# Increasing the lower bounds can facilitate the training as in this example:
predicted_output_data_ = model.predict(input_data).ravel()
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - Bounds")
plt.grid()
plt.legend()
plt.show()

# %%
# Alpha
# ~~~~~
# The ``alpha`` parameter (default: 1e-10),
# often called *nugget effect*,
# is the value added to the diagonal of the training kernel matrix
# to avoid overfitting.
# When ``alpha`` is equal to zero,
# the GP model interpolates the training points
# at which the standard deviation is equal to zero.
# The larger ``alpha`` is, the less interpolating the GP model is.
# For example, we can increase the value to 0.1:
predicted_output_data_1 = predicted_output_data_
model = create_regression_model(
    "GaussianProcessRegressor", training_dataset, bounds=(1e-1, 1e2), alpha=0.1
)
model.learn()
# %%
# and see that the model moves away from the training points:
predicted_output_data_2 = model.predict(input_data).ravel()
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data_1, label="Regression - Alpha(1e-10)")
plt.plot(input_data.ravel(), predicted_output_data_2, label="Regression - Alpha(1e-1)")
plt.grid()
plt.legend()
plt.show()
