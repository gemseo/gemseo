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
"""# Define a parameter space with deterministic and uncertain variables

## Problem

Your analysis mixes deterministic design variables (with fixed bounds)
and uncertain random variables (described by probability distributions).
You need a single object that holds both kinds
and lets you distinguish between them.

## Solution

[ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
extends
[DesignSpace][gemseo.algos.design_space.DesignSpace]
to accommodate random variables alongside deterministic ones.
Use
[add_variable()][gemseo.algos.parameter_space.ParameterSpace.add_variable]
for deterministic variables and
[add_random_variable()][gemseo.algos.parameter_space.ParameterSpace.add_random_variable]
for uncertain ones.
Use
[is_deterministic()][gemseo.algos.parameter_space.ParameterSpace.is_deterministic]
and
[is_uncertain()][gemseo.algos.parameter_space.ParameterSpace.is_uncertain]
to query the type of any variable.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.settings.probability_distributions import SPNormalDistribution_Settings

# %%
# ### 1. Create a parameter space
#
# A [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace] requires no mandatory arguments:
parameter_space = ParameterSpace()

# %%
# ### 2. Add a deterministic variable
#
# Use [add_variable()][gemseo.algos.parameter_space.ParameterSpace.add_variable]
# with lower and upper bounds:
parameter_space.add_variable("x", lower_bound=-2.0, upper_bound=2.0)

# %%
# ### 3. Add an uncertain variable
#
# Use [add_random_variable()][gemseo.algos.parameter_space.ParameterSpace.add_random_variable]
# with a distribution settings object:
parameter_space.add_random_variable(
    "y", SPNormalDistribution_Settings(mu=0.0, sigma=1.0)
)
parameter_space

# %%
# !!! note
#
#     All available distribution settings classes can be imported from
#     [gemseo.settings.probability_distributions][gemseo.settings.probability_distributions].
#     `SP` uses SciPy, `OT` uses OpenTURNS — do not mix both in the same space.
#     See [Probability distributions][]
#     for a full introduction to backends.

# %%
# ### 4. Check variable types
#
# Confirm which variables are deterministic and which are uncertain:
parameter_space.is_deterministic("x"), parameter_space.is_uncertain("y")

# %%
# ## Summary
#
# - [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
#   holds both deterministic and uncertain variables;
# - [add_variable()][gemseo.algos.parameter_space.ParameterSpace.add_variable]
#   adds deterministic variables (bounds);
# - [add_random_variable()][gemseo.algos.parameter_space.ParameterSpace.add_random_variable]
#   adds uncertain variables (distribution settings from
#   [gemseo.settings.probability_distributions][gemseo.settings.probability_distributions]);
# - [is_deterministic()][gemseo.algos.parameter_space.ParameterSpace.is_deterministic]
#   and [is_uncertain()][gemseo.algos.parameter_space.ParameterSpace.is_uncertain]
#   identify the type of each variable.
#
# ## One step further
#
# - To sample the uncertain variables from this space,
#   use [compute_samples()][gemseo.algos.parameter_space.ParameterSpace.compute_samples] —
#   it works the same way as shown in
#   [Define an uncertain space][].
# - To propagate uncertainty through a discipline via DOE,
#   see [Propagate uncertainty through a discipline][].
# - [extract_uncertain_space()][gemseo.algos.parameter_space.ParameterSpace.extract_uncertain_space]
#   restricts the space to random variables only;
#   it is demonstrated in
#   [Propagate uncertainty through a discipline][].
