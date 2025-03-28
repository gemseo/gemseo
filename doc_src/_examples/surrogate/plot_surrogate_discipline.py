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
Create a surrogate discipline
=============================

We want to build an :class:`.Discipline`
based on a regression model approximating the following discipline
with two inputs and two outputs:

- :math:`y_1=1+2x_1+3x_2`
- :math:`y_2=-1-2x_1-3x_2`

over the unit hypercube :math:`[0,1]\\times[0,1]`.
For that,
we use a :class:`.SurrogateDiscipline` relying on an :class:`.BaseRegressor`.
"""

from __future__ import annotations

from numpy import array

from gemseo import configure_logger
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_surrogate
from gemseo import sample_disciplines

# %%
# Import
# ------

configure_logger()


# %%
# Create the discipline to learn
# ------------------------------
# We can implement this analytic discipline by means of the
# :class:`~gemseo.disciplines.analytic.AnalyticDiscipline` class.
expressions = {"y_1": "1+2*x_1+3*x_2", "y_2": "-1-2*x_1-3*x_2"}
discipline = create_discipline(
    "AnalyticDiscipline", name="func", expressions=expressions
)

# %%
# Create the input sampling space
# -------------------------------
# We create the input sampling space by adding the variables one by one.
design_space = create_design_space()
design_space.add_variable("x_1", lower_bound=0.0, upper_bound=1.0)
design_space.add_variable("x_2", lower_bound=0.0, upper_bound=1.0)

# %%
# Create the training dataset
# ---------------------------
# We can build a training dataset
# by sampling the discipline using the :func:`.sample_disciplines`
# with a full factorial design of experiments.
dataset = sample_disciplines(
    [discipline], design_space, ["y_1", "y_2"], algo_name="PYDOE_FULLFACT", n_samples=9
)

# %%
# Create the surrogate discipline
# -------------------------------
# Then, we build the Gaussian process regression model from the dataset and
# displays this model.
model = create_surrogate("GaussianProcessRegressor", data=dataset)

# %%
# Predict output
# --------------
# Once it is built, we can use it for prediction, either with default inputs
model.execute()
# %%
# or with user-defined ones.
model.execute({"x_1": array([1.0]), "x_2": array([2.0])})
