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
"""# Radial basis function (RBF) regression.

An [RBFRegressor][gemseo.machine_learning.regression.model.rbf.RBFRegressor] is an RBF model
based on [SciPy](https://scipy.org).

!!! info "See also"
    You can find more information about RBF models on
    [this wikipedia page](https://en.wikipedia.org/wiki/Radial_basis_function_interpolation).
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from numpy import array

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import sample_disciplines
from gemseo.enum import RBF
from gemseo.machine_learning import create_regression_model

# %%
# ## Problem
#
# You want to build a radial basis function (RBF) regression model from data,
# and assess how well it approximates the underlying function.
#
# In this example,
# you represent the function $f(x)=(6x-2)^2\sin(12x-4)$
# by the [AnalyticDiscipline][gemseo.discipline.analytic.AnalyticDiscipline].
#
# !!! quote "References"
#       Alexander I. J. Forrester, Andras Sobester, and Andy J. Keane.
#       Engineering design via surrogate modelling: a practical guide. Wiley, 2008.
discipline = create_discipline(
    "AnalyticDiscipline",
    {"y": "(6*x-2)**2*sin(12*x-4)"},
    name="f",
)
# %%
# and you seek to approximate it over the input space
input_space = create_design_space()
input_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)

# %%
# To do this,
# you create a training dataset with 6 equispaced points:
training_dataset = sample_disciplines(
    [discipline], input_space, "y", algo_name="PYDOE_FULLFACT", n_samples=6
)

# %%
# ## Basics
#
# ### Training
#
# Then,
# you train an RBF regression model from these samples:
model = create_regression_model("RBFRegressor", training_dataset)
model.learn()

# %%
# ### Prediction
#
# Once it is built,
# you can predict the output value of $f$ at a new input point:
input_value = {"x": array([0.65])}
output_value = model.predict(input_value)
output_value

# %%
# as well as its Jacobian value:
jacobian_value = model.predict_jacobian(input_value)
jacobian_value

# %%
# ### Plotting
#
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
# ## Settings { #rbf-settings }
#
# The [RBFRegressor][gemseo.machine_learning.regression.model.rbf.RBFRegressor] has many options
# defined in the [RBFRegressor_Settings][gemseo.machine_learning.regression.model.rbf_settings.RBFRegressor_Settings] Pydantic model.
#
# ### Kernel
#
# The default RBF is the multiquadric function $-\sqrt{(\epsilon r)^2 + 1}$
# depending on a radius $r$ representing a distance between two points
# and a shape parameter $\epsilon$.
# The RBF can be changed using the `kernel` option
# (type: [RBF][gemseo.machine_learning.regression.model.rbf_settings.RBF]);
# for example, select a Gaussian one:
model = create_regression_model("RBFRegressor", training_dataset, kernel=RBF.GAUSSIAN)
model.learn()
predicted_output_data_g = model.predict(input_data).ravel()
# %%
# or a cubic one:
model = create_regression_model("RBFRegressor", training_dataset, kernel=RBF.CUBIC)
model.learn()
predicted_output_data_c = model.predict(input_data).ravel()
# %%
# You can see that the predictions are different:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_g, label="Regression - Gaussian RBF")
plt.plot(input_data.ravel(), predicted_output_data_c, label="Regression - Cubic RBF")
plt.grid()
plt.legend()
plt.show()

# %%
# ### Epsilon
#
# Some RBFs depend on an `epsilon` parameter,
# namely the shape parameter scaling the radius as $\epsilon r$:
# the greater `epsilon`, the narrower the RBF.
# This is the case of the `"multiquadric"`, `"inverse_multiquadric"`,
# `"inverse_quadratic"` and `"gaussian"` RBFs,
# for which the default value is the reciprocal
# of the average distance between input data.
# For example,
# you can train a first multiquadric RBF model with an `epsilon` set to 2.0
model = create_regression_model("RBFRegressor", training_dataset, epsilon=2.0)
model.learn()
predicted_output_data_1 = model.predict(input_data).ravel()
# %%
# a second one with an `epsilon` set to 1.0:
model = create_regression_model("RBFRegressor", training_dataset, epsilon=1.0)
model.learn()
predicted_output_data_2 = model.predict(input_data).ravel()
# %%
# and a last one with an `epsilon` set to 0.5:
model = create_regression_model("RBFRegressor", training_dataset, epsilon=0.5)
model.learn()
predicted_output_data_3 = model.predict(input_data).ravel()
# %%
# and you see that this parameter represents the regularity of the regression model:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_1, label="Regression - Epsilon(2)")
plt.plot(input_data.ravel(), predicted_output_data_2, label="Regression - Epsilon(1)")
plt.plot(input_data.ravel(), predicted_output_data_3, label="Regression - Epsilon(0.5)")
plt.grid()
plt.legend()
plt.show()

# %%
# ### Smoothing
#
# By default,
# an RBF model interpolates the training points.
# This is parametrized by the `smoothing` option which is set to 0.
# You can increase the smoothness of the model by increasing this value:
model = create_regression_model("RBFRegressor", training_dataset, smoothing=0.1)
model.learn()
predicted_output_data_ = model.predict(input_data).ravel()
# %%
# and you see that the model is not interpolating:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - Smoothing")
plt.grid()
plt.legend()
plt.show()

# %%
# ### Degree
#
# The model adds a low-degree polynomial to the weighted sum of RBFs,
# whose degree is by default the minimum degree required by the kernel.
# You can change it with the `degree` option,
# or remove the polynomial by setting `degree` to `-1`:
model = create_regression_model("RBFRegressor", training_dataset, degree=-1)
model.learn()
predicted_output_data_ = model.predict(input_data).ravel()
# %%
# and you see that both models interpolate the learning points
# but differ in between:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - Degree(-1)")
plt.grid()
plt.legend()
plt.show()

# %%
# ## Thin plate spline (TPS)
#
# TPS regression is a specific case of RBF regression
# where the RBF is the thin plate spline radial basis function
# $(\epsilon r)^2\log(\epsilon r)$.
# The [TPSRegressor][gemseo.machine_learning.regression.model.tps.TPSRegressor] class
# deriving from [RBFRegressor][gemseo.machine_learning.regression.model.rbf.RBFRegressor]
# implements this case:
model = create_regression_model("TPSRegressor", training_dataset)
model.learn()
predicted_output_data_ = model.predict(input_data).ravel()
# %%
# You can see the difference between this model
# and the default multiquadric RBF model:
plt.plot(input_data.ravel(), reference_output_data, label="Reference")
plt.plot(input_data.ravel(), predicted_output_data, label="Regression - Basics")
plt.plot(input_data.ravel(), predicted_output_data_, label="Regression - TPS")
plt.grid()
plt.legend()
plt.show()

# %%
# The [TPSRegressor][gemseo.machine_learning.regression.model.tps.TPSRegressor]
# can be customized with the [TPSRegressor_Settings][gemseo.machine_learning.regression.model.tps_settings.TPSRegressor_Settings].
