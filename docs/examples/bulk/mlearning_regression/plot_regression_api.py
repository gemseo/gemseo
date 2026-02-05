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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# High-level functions.

The [gemseo.machine_learning][gemseo.machine_learning] package includes high-level
functions
to create regression models from model class names.
"""

from __future__ import annotations

from gemseo import create_benchmark_dataset
from gemseo.machine_learning import create_regression_model
from gemseo.machine_learning import get_regression_models
from gemseo.machine_learning import get_regression_options

# %%
# ## Available models
#
# Use the [get_regression_models()][gemseo.machine_learning.get_regression_models]
# to list the available model class names:
get_regression_models()

# %%
# ## Available model options
#
# Use the [get_regression_options()][gemseo.machine_learning.get_regression_options]
# to get the options of a model
# from its class name:
get_regression_options("GaussianProcessRegressor", pretty_print=False)

# %%
#
# !!! info "See also"
#
#     The functions
#     [get_regression_models()][gemseo.machine_learning.get_regression_models] and [
#     get_regression_options()][gemseo.machine_learning.get_regression_options]
#     can be very useful for the developers.
#     As a user,
#     it may be easier to consult [this page][available-regression-models]
#     to find out about the different models and their options.
#
# ## Creation
#
# Given a training dataset, *e.g.*
dataset = create_benchmark_dataset("RosenbrockDataset", opt_naming=False)
# %%
# use the [create_regression_model()][
# gemseo.machine_learning.create_regression_model] function
# to create a clustering model from its class name and settings:
model = create_regression_model("RBFRegressor", data=dataset)
model.learn()
