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
Random forest
=============

A :class:`.RandomForestRegressor` is a random forest model
based on `scikit-learn <https://scikit-learn.org/>`__.
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
# we train an random forest regression model from these samples:
model = create_regression_model("RandomForestRegressor", training_dataset)
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
    print("The derivatives are not available for RandomForestRegressor.")

# %%
# Plotting
# ~~~~~~~~
# You can see that the random forest model is pretty good on the left,
# but bad on the right:
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
# Number of estimators
# ~~~~~~~~~~~~~~~~~~~~
# The main hyperparameter of random forest regression is
# the number of trees in the forest (default: 100).
# Here is a comparison when increasing and decreasing this number:
model = create_regression_model(
    "RandomForestRegressor", training_dataset, n_estimators=10
)
model.learn()
predicted_output_data_1 = model.predict(input_data).ravel()
model = create_regression_model(
    "RandomForestRegressor", training_dataset, n_estimators=1000
)
model.learn()
predicted_output_data_2 = model.predict(input_data).ravel()
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_1, label="Regression - 10 trees")
plt.plot(input_data.ravel(), predicted_output_data_2, label="Regression - 1000 trees")
plt.grid()
plt.legend()
plt.show()

# %%
# Others
# ------
# The ``RandomForestRegressor`` class of scikit-learn has a lot of settings
# (`read more <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`__),
# and we have chosen to exhibit only ``n_estimators``.
# However,
# any argument of ``RandomForestRegressor`` can be set
# using the dictionary ``parameters``.
# For example,
# we can impose a minimum of two samples per leaf:
model = create_regression_model(
    "RandomForestRegressor", training_dataset, parameters={"min_samples_leaf": 2}
)
model.learn()
predicted_output_data_ = model.predict(input_data).ravel()
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - 2 samples")
plt.grid()
plt.legend()
plt.show()
