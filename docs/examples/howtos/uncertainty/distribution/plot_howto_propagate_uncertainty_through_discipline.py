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
"""# Propagate uncertainty through a discipline

## Problem

You have a parameter space that mixes deterministic design variables
and uncertain random variables,
and you want to evaluate a discipline at many sampled input points
to study how uncertainty in the inputs propagates to the outputs.

## Solution

[sample_disciplines()][gemseo.sample_disciplines]
runs a Design of Experiments (DOE) over a
[ParameterSpace][gemseo.space.parameter.ParameterSpace]
and returns an [IODataset][gemseo.dataset.io_dataset.IODataset]
that you can then visualize or pass to a statistics tool.
Use
[extract_uncertain_space()][gemseo.space.parameter.ParameterSpace.extract_uncertain_space]
to restrict the DOE to the random variables only.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import sample_disciplines
from gemseo.discipline import AnalyticDiscipline
from gemseo.post.dataset import PairPlot
from gemseo.space import ParameterSpace
from gemseo.uncertainty.distribution import SPNormalDistribution_Settings

# %%
# ### 1. Set up the discipline and parameter space
#
# Create a simple analytic discipline:
discipline = AnalyticDiscipline({"z": "x+y"})

# %%
# Build a parameter space with one deterministic variable `x`
# and one uncertain variable `y`:
parameter_space = ParameterSpace()
parameter_space.add_variable("x", lower_bound=-2.0, upper_bound=2.0)
parameter_space.add_random_variable(
    "y", SPNormalDistribution_Settings(mu=0.0, sigma=1.0)
)
parameter_space

# %%
# ### 2. Sample the discipline over the full parameter space
#
# Run a Latin Hypercube Sampling (LHS) DOE over the mixed parameter space
# and collect inputs and outputs in an
# [IODataset][gemseo.dataset.io_dataset.IODataset]:
dataset = sample_disciplines(
    [discipline], parameter_space, "z", algo_name="PYDOE_LHS", n_samples=100
)
dataset.describe()

# %%
# ### 3. Visualize the input-output samples
#
# A pair plot shows marginal histograms on the diagonal
# and scatter plots for each pair of variables off the diagonal:
PairPlot(dataset).execute(save=False, show=True)

# %%
# ### 4. Restrict propagation to uncertain variables only
#
# Extract the uncertain subspace to sample only the random variables,
# keeping deterministic variables at their nominal values:
uncertain_space = parameter_space.extract_uncertain_space()
dataset_uncertain = sample_disciplines(
    [discipline], uncertain_space, "z", algo_name="PYDOE_LHS", n_samples=100
)
dataset_uncertain.describe()

# %%
# ## Summary
#
# - [sample_disciplines()][gemseo.sample_disciplines]
#   runs a DOE over a [ParameterSpace][gemseo.space.parameter.ParameterSpace]
#   and returns an [IODataset][gemseo.dataset.io_dataset.IODataset];
# - pass the full mixed space to sample both deterministic and uncertain variables together,
#   or pass the uncertain subspace to fix deterministic variables at their nominal values;
# - [extract_uncertain_space()][gemseo.space.parameter.ParameterSpace.extract_uncertain_space]
#   restricts the space to its random variables;
# - [PairPlot][gemseo.post.dataset.pair_plot.PairPlot]
#   visualizes the joint distribution of inputs and outputs.
#
# ## One step further
#
# To compute mean, variance, or quantiles from this dataset,
# see [Compute empirical statistics from a dataset][].
