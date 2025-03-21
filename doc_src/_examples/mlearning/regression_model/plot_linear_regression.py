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
Linear regression
=================

A :class:`.LinearRegressor` is a linear regression model
based on `scikit-learn <https://scikit-learn.org/>`__.

.. seealso::
   You can find more information about building linear models with scikit-learn on
   `this page <https://scikit-learn.org/stable/modules/linear_model.html>`__.
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
# we train a linear regression model from these samples:
model = create_regression_model("LinearRegressor", training_dataset)
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
# you can see that the linear model is no good at all here:
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
# The :class:`.LinearRegressor` has many options
# defined in the :class:`.LinearRegressor_Settings` Pydantic model.
#
# Intercept
# ~~~~~~~~~~
# By default,
# the linear model is of the form :math:`a_0+a_1x_1+\ldots+a_dx_d`.
# You can set the option ``fit_intercept`` to ``False``
# if you want a linear model of the form :math:`a_1x_1+\ldots+a_dx_d`:
model = create_regression_model(
    "LinearRegressor", training_dataset, fit_intercept=False, transformer={}
)
model.learn()
# %%
# .. warning::
#    This notion applies in the space of transformed variables.
#    This is the reason why
#    we removed the default transformers by setting ``transformer`` to ``{}``.
#
# We can see the impact of this option in the following visualization:
predicted_output_data_ = model.predict(input_data).ravel()
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - No intercept")
plt.grid()
plt.legend()
plt.show()

# %%
# Regularization
# ~~~~~~~~~~~~~~
# When the number of samples is small relative to the input dimension,
# regularization techniques can save you from overfitting
# (a model that is very good at learning but bad at generalization).
# The ``penalty_level`` option is a positive real number
# defining the degree of regularization (default: no regularization).
# By default,
# the regularization technique is the ridge penalty (l2 regularization).
# The technique can be replaced by the lasso penalty (l1 regularization)
# by setting the ``l2_penalty_ratio`` option to ``0.0``.
# When ``l2_penalty_ratio`` is between 0 and 1,
# the regularization technique is the elastic net penalty,
# *i.e.* a linear combination of ridge and lasso penalty
# parametrized by this ``l2_penalty_ratio``.
#
# For example,
# we can use the ridge penalty with a level of 1.2
model = create_regression_model("LinearRegressor", training_dataset, penalty_level=1.2)
model.learn()
predicted_output_data_ = model.predict(input_data).ravel()
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - Ridge(1.2)")
plt.grid()
plt.legend()
plt.show()
# %%
# We can see that the coefficient of the linear model is lower due to the penalty.
#
# .. note::
#    In the case of a model with many inputs,
#    we could have used the lasso penalty
#    and seen that some coefficients would have been set to zero.
