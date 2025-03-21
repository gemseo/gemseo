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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Polynomial regression
=====================

A :class:`.PolynomialRegressor` is a polynomial regression model
based on a :class:`.LinearRegressor`.
This design choice was made
because a polynomial regression model is a generalized linear model
whose basis functions are monomials.
Thus,
a :class:`.PolynomialRegressor` benefits
from the same settings as :class:`.LinearRegressor`:
offset can be set to zero and regularization techniques can be used.

.. seealso::
   You will find more information about these settings
   in :ref:`the example about the linear regression model <sphx_glr_examples_mlearning_regression_model_plot_polynomial_regression.py>`.
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from numpy import array

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
# we train a polynomial regression model with a degree of 2 (default)
# from these samples:
model = create_regression_model("PolynomialRegressor", training_dataset)
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
# as well as its Jacobian value:
jacobian_value = model.predict_jacobian(input_value)
jacobian_value

# %%
# Plotting
# ~~~~~~~~
# Of course,
# you can see that the quadratic model is no good at all here:
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
# The :class:`.PolynomialRegressor` has many options
# defined in the :class:`.PolynomialRegressor_Settings` Pydantic model.
# Most of them are presented
# in :ref:`the example about the linear regression model <sphx_glr_examples_mlearning_regression_model_plot_polynomial_regression.py>`.
#
# The only one we will look at here is the degree of the polynomial regression model.
# This information can be set with the ``degree`` keyword.
# For example,
# we can use a cubic regression model instead of a quadratic one:
model = create_regression_model("PolynomialRegressor", training_dataset, degree=3)
model.learn()
# %%
# and see that this model seems to be better:
predicted_output_data_ = model.predict(input_data).ravel()
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - Degree(3)")
plt.grid()
plt.legend()
plt.show()
