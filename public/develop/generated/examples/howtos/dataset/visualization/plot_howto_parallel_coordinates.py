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
"""# Parallel coordinates chart

## Problem

Visualise multiple variables simultaneously across many samples,
and identify patterns or clusters by class.

## Solution

Use [ParallelCoordinates][gemseo.post.dataset.parallel_coordinates.ParallelCoordinates],
a.k.a. cobweb plot, where each sample is represented as a polyline
whose nodes are positioned along one axis per variable.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_benchmark_dataset
from gemseo.post.dataset.parallel_coordinates import ParallelCoordinates
from gemseo.post.dataset.parallel_coordinates_settings import (
    ParallelCoordinates_Settings,
)

# %%
# ### 1. Build the dataset
#
iris = create_benchmark_dataset("IrisDataset")

# %%
# ### 2. Plot the parallel coordinates chart
#
# Pass a `classifier` variable name to colour each polyline by class:
ParallelCoordinates(iris, ParallelCoordinates_Settings(classifier="specy")).execute(
    save=False, show=True
)

# %%
# ## Summary
#
# [ParallelCoordinates][gemseo.post.dataset.parallel_coordinates.ParallelCoordinates]
# displays each sample as a polyline across all variable axes.
# Use the `classifier` argument to colour lines by a categorical variable
# and reveal cluster structure.
