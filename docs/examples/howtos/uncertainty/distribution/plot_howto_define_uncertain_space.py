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
"""# Define an uncertain space

## Problem

You want to define a space of uncertain variables — for example
to run an uncertainty propagation or a sensitivity analysis —
without any deterministic variable.

## Solution

[ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
used with only
[add_random_variable()][gemseo.algos.parameter_space.ParameterSpace.add_random_variable]
calls acts as a pure uncertain space.
Pass distribution settings objects (importable from
[gemseo.settings.probability_distributions][gemseo.settings.probability_distributions])
to describe each variable.
Then sample from it directly or pass it to
[sample_disciplines()][gemseo.sample_disciplines]
to propagate uncertainty through a discipline.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.settings.probability_distributions import SPNormalDistribution_Settings
from gemseo.settings.probability_distributions import SPUniformDistribution_Settings

# %%
# ### 1. Create an uncertain space
#
# A [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
# requires no mandatory arguments:
uncertain_space = ParameterSpace()

# %%
# ### 2. Add uncertain variables
#
# Use [add_random_variable()][gemseo.algos.parameter_space.ParameterSpace.add_random_variable]
# with a distribution settings object.
# Here `x` follows a standard normal distribution:
uncertain_space.add_random_variable("x", SPNormalDistribution_Settings())

# %%
# And `y` a random vector whose components are independent
# and follow a uniform distribution on [-1, 1]:
uncertain_space.add_random_variable(
    "y", SPUniformDistribution_Settings(minimum=-1.0, maximum=1.0), size=2
)

# %%
# ### 3. Add a vector with mixed distributions
#
# When components follow different distributions,
# use [add_random_vector()][gemseo.algos.parameter_space.ParameterSpace.add_random_vector]
# with one settings object per component:
uncertain_space.add_random_vector(
    "z",
    SPUniformDistribution_Settings(minimum=-1.0, maximum=1.0),
    SPNormalDistribution_Settings(mu=0.5, sigma=1.8),
)
uncertain_space

# %%
# !!! note
#     `SP` and `OT` prefix distribution settings classes for SciPy and OpenTURNS respectively.
#     Do not mix both prefixes in the same space.
#     See [Probability distributions][]
#     for a full introduction to backends.
#
# ### 4. List uncertain variables
uncertain_space.uncertain_variables

# %%
# ### 5. Query per-variable statistics
#
# Numerical range of `x` (difference between finite numerical bounds):
uncertain_space.get_range("x")

# %%
# Mathematical support of `x` (exact bounds of the probability distribution):
uncertain_space.get_support("x")

# %%
# ### 6. Sample from the space
#
# Draw 5 samples as a concatenated NumPy array:
uncertain_space.compute_samples(n_samples=5)

# %%
# Or as a dictionary mapping variable names to arrays:
uncertain_space.compute_samples(n_samples=5, as_dict=True)

# %%
# ## Summary
#
# - [ParameterSpace][gemseo.algos.parameter_space.ParameterSpace]
#   with only [add_random_variable()][gemseo.algos.parameter_space.ParameterSpace.add_random_variable]
#   calls defines a pure uncertain space;
# - distribution settings classes are importable from
#   [gemseo.settings.probability_distributions][gemseo.settings.probability_distributions]
#   (names prefixed with `SP` use SciPy, `OT` use OpenTURNS —
#   do not mix both prefixes in the same space);
# - [get_range()][gemseo.algos.parameter_space.ParameterSpace.get_range]
#   and [get_support()][gemseo.algos.parameter_space.ParameterSpace.get_support]
#   return the numerical range and mathematical support of each uncertain variable;
# - [compute_samples()][gemseo.algos.parameter_space.ParameterSpace.compute_samples]
#   draws random samples from the joint distribution.
#
# ## One step further
#
# To propagate uncertainty through a discipline using these samples,
# see [Propagate uncertainty through a discipline][].
