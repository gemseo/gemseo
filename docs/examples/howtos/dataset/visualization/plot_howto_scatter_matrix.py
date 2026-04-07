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
"""# Scatter matrix

## Problem

Visualise pairwise relationships between all variables in a dataset at once,
with per-class colour coding.

## Solution

Use [ScatterMatrix][gemseo.post.dataset.scatter_plot_matrix.ScatterMatrix],
which renders all pairwise scatter plots in off-diagonal blocks
and per-variable distribution estimates (histogram or KDE) on the diagonal.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_benchmark_dataset
from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix
from gemseo.post.dataset.scatter_plot_matrix_settings import ScatterMatrix_Settings

# %%
# ### 1. Build the dataset
#
iris = create_benchmark_dataset("IrisDataset")

# %%
# ### 2. Plot the scatter matrix
#
# Pass a `classifier` variable name to colour the points by class:
ScatterMatrix(iris, ScatterMatrix_Settings(classifier="specy")).execute(
    save=False, show=True
)

# %%
# ## Summary
#
# [ScatterMatrix][gemseo.post.dataset.scatter_plot_matrix.ScatterMatrix]
# provides an overview of all pairwise relationships in a single figure.
# Use the `classifier` argument to colour points by a categorical variable.
