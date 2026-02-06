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
"""# Use a surrogate model

## Problem

Given a training dataset,
how can I create a surrogate model and use it as a discipline?

## Solution

Instantiate
the [SurrogateDiscipline][gemseo.disciplines.surrogate.SurrogateDiscipline] class
from regression model settings and an [IODataset][gemseo.datasets.io_dataset.IODataset].

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo import sample_disciplines
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.disciplines.surrogate import SurrogateDiscipline
from gemseo.machine_learning.regression.models.rbf_settings import RBFRegressor_Settings
from gemseo.problems.uncertainty.wing_weight.discipline import WingWeightDiscipline
from gemseo.problems.uncertainty.wing_weight.uncertain_space import (
    WingWeightUncertainSpace,
)

# %%
# ### 1. Define the reference model.
#
# In this how-to guide,
# we consider the wing weight problem
# defined in [this page][gemseo.problems.uncertainty.wing_weight].

discipline = WingWeightDiscipline()
input_space = WingWeightUncertainSpace()

# %%
# ### 2. Build a small training dataset
#
# We generate $3 \times d$ training samples
# using optimized Latin hypercube sampling strategy.

training_dataset = sample_disciplines(
    [discipline],
    input_space,
    ["Ww"],
    algo_settings_model=OT_OPT_LHS_Settings(n_samples=input_space.dimension * 3),
)

# %%
# ### 3. Create a surrogate discipline
#
# We create a discipline based on a regressor trained from this training dataset.

discipline = SurrogateDiscipline.from_settings(
    RBFRegressor_Settings(), training_dataset
)
discipline

# %%
# This discipline could also be created directly from a regressor,
# using `discipline = SurrogateDiscipline(regressor)`.
#
# Note that
# the surrogate discipline automatically scales the input and output variables
# between 0 and 1. You can change this behavior using the `transformer` option.
#
# The regressor can be easily accessed.
discipline.regressor

# %%
# ### 4. Evaluate the surrogate quality
#
# The surrogate discipline as a specific method for assessing its quality.

r2 = discipline.get_error_measure("R2Measure")
r2.compute_learning_measure()

# %%
# ### 5. Evaluate the surrogate discipline
#
# #### Default input values
discipline.execute()
discipline.get_output_data()

# %%
# !!! note
#
#     Its default input values correspond to
#     the center of the smallest subset of the input space
#     containing the training dataset.
#
# #### Custom input values
discipline.execute({"Nz": array([4.25]), "Sw": array([175])})
discipline.get_output_data()

# %%
# #### Derivatives
discipline.linearize({"Nz": array([4.25]), "Sw": array([175])})

# %%
# ## Summary
#
# A surrogate model can be created as a discipline
# from an [IODataset][gemseo.datasets.io_dataset.IODataset] and regressor settings
# using [SurrogateDiscipline][gemseo.disciplines.surrogate.SurrogateDiscipline].
