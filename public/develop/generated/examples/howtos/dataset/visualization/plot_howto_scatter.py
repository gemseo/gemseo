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
"""# Scatter chart

## Problem

You want to visualize the relationship between two variables `x` and `y` as individual points,
with optional per-point color coding.

## Solution

Use [Scatter][gemseo.post.dataset.scatter.Scatter],
which renders a scatter plot of `y` against `x`
and supports per-point color assignment.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import linspace
from numpy import pi
from numpy import sin

from gemseo.dataset import Dataset
from gemseo.post.dataset import Scatter
from gemseo.post.dataset.scatter_settings import Scatter_Settings

# %%
# ### 1. Build the dataset
#
inputs = linspace(0, 1, 20)[:, None]
outputs = sin(2 * pi * inputs)

dataset = Dataset()
dataset.add_variable("x", inputs, "inputs")
dataset.add_variable("y", outputs, "outputs")

# %%
# ### 2. Define a per-point color
#
# Each point is colored according to whether its output value exceeds 0.5 in magnitude:
color = ["b" if abs(output) > 0.5 else "r" for output in outputs]

# %%
# ### 3. Plot the scatter chart
#
plot = Scatter(dataset, Scatter_Settings(x="x", y="y"))
plot.color = color
plot.execute(save=False, show=True)

# %%
# ## Summary
#
# Pass the variable names as `x` and `y` to [Scatter][gemseo.post.dataset.scatter.Scatter]
# and set `color` to a list of color values to highlight specific points.
