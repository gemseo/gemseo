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

You have a random space describing the random inputs of a discipline
and you want to evaluate this discipline at many sampled input points
to study how uncertainty in the inputs propagates to the outputs.

## Solution

[sample_disciplines()][gemseo.sample_disciplines]
runs a Design of Experiments (DOE) over a
[RandomSpace][gemseo.space.random.RandomSpace]
and returns an [IODataset][gemseo.dataset.io_dataset.IODataset]
that you can then visualize or pass to a statistics tool.
The DOE is generated in the unit hypercube
and mapped to the random space by its iso-probabilistic transformation,
so the samples follow the probability distributions of the random variables.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_random_space
from gemseo import sample_disciplines
from gemseo.discipline import AnalyticDiscipline
from gemseo.doe import PYDOE_LHS_Settings
from gemseo.post.dataset import PairPlot
from gemseo.uncertainty.distribution import SPNormalDistribution_Settings
from gemseo.uncertainty.distribution import SPUniformDistribution_Settings

# %%
# ### Prerequisites
#
# This how-to needs a discipline and a random space.
#
# Create a simple analytic discipline:
discipline = AnalyticDiscipline({"z": "x+y"})

# %%
# Build a random space with two random variables:
random_space = create_random_space()
random_space.add_variable(
    "x", SPUniformDistribution_Settings(minimum=-2.0, maximum=2.0)
)
random_space.add_variable("y", SPNormalDistribution_Settings(mu=0.0, sigma=1.0))
random_space

# %%
# ### 1. Sample the discipline over the random space
#
# Run a Latin Hypercube Sampling (LHS) DOE over the random space
# and collect inputs and outputs in an
# [IODataset][gemseo.dataset.io_dataset.IODataset]:
dataset = sample_disciplines(
    [discipline],
    random_space,
    "z",
    algo_settings_model=PYDOE_LHS_Settings(n_samples=100),
)
dataset.describe()

# %%
# ### 2. Visualize the input-output samples
#
# A pair plot shows marginal histograms on the diagonal
# and scatter plots for each pair of variables off the diagonal:
PairPlot(dataset).execute(save=False, show=True)

# %%
# ## Summary
#
# - [sample_disciplines()][gemseo.sample_disciplines]
#   runs a DOE over a [RandomSpace][gemseo.space.random.RandomSpace]
#   and returns an [IODataset][gemseo.dataset.io_dataset.IODataset];
# - the samples follow the probability distributions of the random variables,
#   because the DOE is mapped from the unit hypercube
#   by the iso-probabilistic transformation of the space;
# - [PairPlot][gemseo.post.dataset.pair_plot.PairPlot]
#   visualizes the joint distribution of inputs and outputs.
#
