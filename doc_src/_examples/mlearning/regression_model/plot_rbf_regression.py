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
Radial basis function (RBF) regression
======================================

An :class:`.RBFRegressor` is an RBF model
based on `SciPy <https://scipy.org/>`__.

.. seealso::
   You can find more information about RBF models on
   `this wikipedia page <https://en.wikipedia.org/wiki/Radial_basis_function_interpolation>`__.
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from numpy import array

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import sample_disciplines
from gemseo.mlearning import create_regression_model
from gemseo.mlearning.regression.algos.rbf_settings import RBF

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
# we train an RBF regression model from these samples:
model = create_regression_model("RBFRegressor", training_dataset)
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
# You can see that the RBF model is pretty good on the right, but bad on the left:
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
# The :class:`.RBFRegressor` has many options
# defined in the :class:`.RBFRegressor_Settings` Pydantic model.
#
# Function
# ~~~~~~~~
# The default RBF is the multiquadratic function :math:`\sqrt{(r/\epsilon)^2 + 1}`
# depending on a radius :math:`r` representing a distance between two points
# and an adjustable constant :math:`\epsilon`.
# The RBF can be changed using the ``function`` option,
# which can be either an :class:`.RBF`:
model = create_regression_model("RBFRegressor", training_dataset, function=RBF.GAUSSIAN)
model.learn()
predicted_output_data_g = model.predict(input_data).ravel()


# %%
# or a Python function:
def rbf(self, r: float) -> float:
    """Evaluate a cubic RBF.

    An RBF must take 2 arguments, namely ``(self, r)``.

    Args:
        r: The radius.

    Returns:
        The RBF value.
    """
    return r**3


model = create_regression_model("RBFRegressor", training_dataset, function=rbf)
model.learn()
predicted_output_data_c = model.predict(input_data).ravel()
# %%
# We can see that the predictions are different:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_g, label="Regression - Gaussian RBF")
plt.plot(input_data.ravel(), predicted_output_data_c, label="Regression - Cubic RBF")
plt.grid()
plt.legend()
plt.show()

# %%
# Epsilon
# ~~~~~~~
# Some RBFs depend on an ``epsilon`` parameter
# whose default value is the average distance between input data.
# This is the case of ``"multiquadric"``, ``"gaussian"`` and ``"inverse"`` RBFs.
# For example,
# we can train a first multiquadric RBF model with an ``epsilon`` set to 0.5
model = create_regression_model("RBFRegressor", training_dataset, epsilon=0.5)
model.learn()
predicted_output_data_1 = model.predict(input_data).ravel()
# %%
# a second one with an ``epsilon`` set to 1.0:
model = create_regression_model("RBFRegressor", training_dataset, epsilon=1.0)
model.learn()
predicted_output_data_2 = model.predict(input_data).ravel()
# %%
# and a last one with an ``epsilon`` set to 2.0:
model = create_regression_model("RBFRegressor", training_dataset, epsilon=2.0)
model.learn()
predicted_output_data_3 = model.predict(input_data).ravel()
# %%
# and see that this parameter represents the regularity of the regression model:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_1, label="Regression - Epsilon(0.5)")
plt.plot(input_data.ravel(), predicted_output_data_2, label="Regression - Epsilon(1)")
plt.plot(input_data.ravel(), predicted_output_data_3, label="Regression - Epsilon(2)")
plt.grid()
plt.legend()
plt.show()

# %%
# Smooth
# ~~~~~~
# By default,
# an RBF model interpolates the training points.
# This is parametrized by the ``smooth`` option which is set to 0.
# We can increase the smoothness of the model by increasing this value:
model = create_regression_model("RBFRegressor", training_dataset, smooth=0.1)
model.learn()
predicted_output_data_ = model.predict(input_data).ravel()
# %%
# and see that the model is not interpolating:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - Smooth")
plt.grid()
plt.legend()
plt.show()

# %%
# Thin plate spline (TPS)
# -----------------------
# TPS regression is a specific case of RBF regression
# where the RBF is the thin plate radial basis function for :math:`r^2\log(r)`.
# The :class:`.TPSRegressor` class
# deriving from :class:`.RBFRegressor`
# implements this case:
model = create_regression_model("TPSRegressor", training_dataset)
model.learn()
predicted_output_data_ = model.predict(input_data).ravel()
# %%
# We can see that the difference between this model
# and the default multiquadric RBF model:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - TPS")
plt.grid()
plt.legend()
plt.show()

# %%
# The :class:`.TPSRegressor` can be customized with the :class:`.TPSRegressor_Settings`.
