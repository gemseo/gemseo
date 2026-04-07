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
"""# Andrews curves

## Problem

Visualise high-dimensional data and reveal cluster structure
by representing each sample as a smooth curve.

## Solution

Use [AndrewsCurves][gemseo.post.dataset.andrews_curves.AndrewsCurves],
a smooth alternative to parallel coordinates where each sample is mapped
to a Fourier series curve; samples from the same class tend to produce
similar curves, making clusters visible.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_benchmark_dataset
from gemseo.post.dataset.andrews_curves import AndrewsCurves
from gemseo.post.dataset.andrews_curves_settings import AndrewsCurves_Settings

# %%
# ### 1. Build the dataset
#
iris = create_benchmark_dataset("IrisDataset")

# %%
# ### 2. Plot the Andrews curves
#
# Pass the classifier variable name to colour each curve by class:
AndrewsCurves(iris, AndrewsCurves_Settings(classifier="specy")).execute(
    save=False, show=True
)

# %%
# ## Summary
#
# [AndrewsCurves][gemseo.post.dataset.andrews_curves.AndrewsCurves] maps each sample
# to a smooth Fourier curve, coloured by the classifier variable.
# Overlapping curves of the same colour indicate a cluster structure in the data.
