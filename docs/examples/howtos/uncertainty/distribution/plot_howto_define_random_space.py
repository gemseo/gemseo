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
"""# Define a random space

## Problem

You want to define a space of uncertain variables — for example
to run an uncertainty propagation or a sensitivity analysis —
without any deterministic variable.

## Solution

Use a [RandomSpace][gemseo.space.random.RandomSpace],
which can be created with
[create_random_space()][gemseo.create_random_space].
Pass distribution settings objects (importable from
[gemseo.uncertainty.distribution][gemseo.uncertainty.distribution])
to describe each variable.
Then sample from it directly or pass it to
[sample_disciplines()][gemseo.sample_disciplines]
to propagate uncertainty through a discipline.

## Step-by-step guide
"""

from __future__ import annotations

from openturns import CorrelationMatrix
from openturns import NormalCopula

from gemseo import create_random_space
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from gemseo.uncertainty.distribution import OTUniformDistribution_Settings
from gemseo.uncertainty.distribution import SPNormalDistribution_Settings
from gemseo.uncertainty.distribution import SPUniformDistribution_Settings

# %%
# ### 1. Create a random space
#
# A [RandomSpace][gemseo.space.random.RandomSpace]
# requires no mandatory arguments:
random_space = create_random_space()

# %%
# ### 2. Add uncertain variables
#
# Use [add_variable()][gemseo.space.random.RandomSpace.add_variable]
# with a distribution settings object.
# Here `x` follows a standard normal distribution:
random_space.add_variable("x", SPNormalDistribution_Settings())

# %%
# And `y` a random vector whose components are independent
# and follow a uniform distribution on [-1, 1],
# using one settings object per component:
random_space.add_variable(
    "y", *[SPUniformDistribution_Settings(minimum=-1.0, maximum=1.0)] * 2
)

# %%
# ### 3. Add a vector with mixed distributions
#
# When components follow different distributions,
# pass their settings objects in order:
random_space.add_variable(
    "z",
    SPUniformDistribution_Settings(minimum=-1.0, maximum=1.0),
    SPNormalDistribution_Settings(mu=0.5, sigma=1.8),
)
random_space

# %%
# !!! note
#     `SP` and `OT` prefix distribution settings classes for SciPy and OpenTURNS respectively.
#     Do not mix both prefixes in the same space.
#     See [Probability distributions][]
#     for a full introduction to backends.
#
# ### 4. List uncertain variables
list(random_space.variables)

# %%
# ### 5. Query per-variable statistics
#
# All the probabilistic information is read through the registry
# [variables][gemseo.space.base.BaseVariableSpace.variables],
# which maps a name to a random vector,
# then through the joint probability distribution of this random vector.
# This registry is read-only:
# the random space is modified with its methods,
# e.g. `add_variable`, `add_copula` and `remove_variable`.
#
# The joint probability distribution of the components of `x`:
random_space.variables["x"].distribution

# %%
# Numerical range of `x` (difference between finite numerical bounds):
random_space.variables["x"].distribution.range

# %%
# Mathematical support of `x` (exact bounds of the probability distribution):
random_space.variables["x"].distribution.support

# %%
# And the joint probability distribution of the whole random space:
random_space.variables.distribution

# %%
# ### 6. Sample from the space
#
# Draw 5 samples as a concatenated NumPy array:
random_space.compute_samples(n_samples=5)

# %%
# Or as a dictionary mapping variable names to arrays, one row per sample:
random_space.compute_samples(n_samples=5, as_dict=True)

# %%
# ### 7. Add a dependency structure
#
# The random variables of a random space are independent by default.
# [add_copula()][gemseo.space.random.RandomSpace.add_copula]
# links some of them with a copula,
# which is the dependency structure of their joint probability distribution.
#
# !!! note
#
#     Copulas require the OpenTURNS backend:
#     adding one to a space built from `SP` settings raises
#     `SPJointDistribution does not support dependent variables.`
#     The space below is therefore defined with `OT` settings.
#
correlated_space = create_random_space()
correlated_space.add_variable(
    "x", OTUniformDistribution_Settings(minimum=0.0, maximum=1.0)
)
correlated_space.add_variable("y", OTNormalDistribution_Settings(mu=1.0, sigma=2.0))
# %%
# The dimension of the copula is the total size of the random variables it covers,
# here two scalar random variables:
R = CorrelationMatrix(2)
R[0, 1] = 0.25
R[1, 0] = 0.25
correlated_space.add_copula(("x", "y"), NormalCopula(R))
# %%
# The `add_copula()` method can be called multiple times.
# A copula covering a single random vector takes its name alone:
correlated_space.add_variable(
    "z",
    OTNormalDistribution_Settings(mu=1.0, sigma=2.0),
    OTNormalDistribution_Settings(mu=1.0, sigma=2.0),
)
R = CorrelationMatrix(2)
R[0, 1] = 0.66
R[1, 0] = 0.66
correlated_space.add_copula("z", NormalCopula(R))
correlated_space

# %%
# The copula is rendered under the table,
# as it relates several random variables.
# It is read back from the registry of the variables,
# in the very order in which `add_copula` takes them,
# so that the dependency structure of a space can be copied
# with `for names, copula in space.variables.copulas: other.add_copula(names, copula)`:
correlated_space.variables.copulas

# %%
# The samples now reflect the dependency between `"x"` and `"y"`:
correlated_space.compute_samples(n_samples=5, as_dict=True)

# %%
# ## Summary
#
# - [RandomSpace][gemseo.space.random.RandomSpace],
#   created with [create_random_space()][gemseo.create_random_space],
#   defines a space containing only random variables;
#   it has neither bounds setters, nor current value, nor normalization,
#   unlike a [DesignSpace][gemseo.space.design.DesignSpace];
# - distribution settings classes are importable from
#   [gemseo.uncertainty.distribution][gemseo.uncertainty.distribution]
#   (names prefixed with `SP` use SciPy, `OT` use OpenTURNS —
#   do not mix both prefixes in the same space);
# - the read-only registry
#   [variables][gemseo.space.base.BaseVariableSpace.variables]
#   gives access to the `range`, `support`, `mean`, `standard_deviation`
#   and `distribution` of each uncertain variable,
#   as well as to the joint distribution of the space;
# - [compute_samples()][gemseo.space.random.RandomSpace.compute_samples]
#   draws random samples from the joint distribution;
# - [add_copula()][gemseo.space.random.RandomSpace.add_copula]
#   makes some random variables dependent,
#   with the OpenTURNS backend only.
#
# ## One step further
#
# To propagate uncertainty through a discipline using these samples,
# see [Propagate uncertainty through a discipline][].
