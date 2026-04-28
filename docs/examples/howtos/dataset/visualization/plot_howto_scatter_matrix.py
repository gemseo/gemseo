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
"""# Pair plot

## Problem

You have a dataset with multiple variables
and want to visualize all pairwise relationships at once,
with different colors for each class.

## Solution

Use [PairPlot][gemseo.post.dataset.pair_plot.PairPlot],
which renders all pairwise scatter plots in off-diagonal blocks
and per-variable distribution estimates (histogram or KDE) on the diagonal.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_benchmark_dataset
from gemseo.post.dataset.pair_plot import PairPlot
from gemseo.post.dataset.pair_plot_settings import PairPlot_Settings

# %%
# ### 1. Build the dataset
#
iris = create_benchmark_dataset("IrisDataset")

# %%
# ### 2. Plot the pair plot
#
PairPlot(iris, PairPlot_Settings()).execute(save=False, show=True)

# %%
# ### 3. Plot the pair plot with options
#
# Color the dots based on the value of a variable:
PairPlot(iris, PairPlot_Settings(classifier="specy")).execute(save=False, show=True)

# %%
# Use kernel density estimators
# instead of histograms on the diagonal and scatter plots for the off-diagonal cells:
PairPlot(iris, PairPlot_Settings(use_scatter=False, use_kde=True)).execute(
    save=False, show=True
)

# %%
# Use the component-wise normalized ranks on the upper part
# and raw data on the lower part:
PairPlot(iris, PairPlot_Settings(use_ranks=True)).execute(save=False, show=True)

# %%
# ## Summary
#
# [PairPlot][gemseo.post.dataset.pair_plot.PairPlot]
# provides an overview of all pairwise relationships in a single figure.
# Use the `classifier` argument to colour points by a categorical variable.
