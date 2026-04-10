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
"""# A scalar output against two inputs

## Problem

Visualise how a scalar quantity `z` varies over a 2D input space `(x, y)`.

## Solution

Use [ZvsXY][gemseo.post.dataset.zvsxy.ZvsXY],
which renders a scatter or surface plot of `z` as a function of two input components.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo import create_benchmark_dataset
from gemseo.post.dataset.zvsxy import ZvsXY
from gemseo.post.dataset.zvsxy_settings import ZvsXY_Settings

# %%
# ### 1. Build the dataset
#
dataset = create_benchmark_dataset("RosenbrockDataset")

# %%
# ### 2. Plot z vs x and y
#
plot = ZvsXY(dataset, ZvsXY_Settings(x=("x", 0), y=("x", 1), z="rosen"))
plot.colormap = "viridis"
plot.execute(save=False, show=True)

# %%
# ## Summary
#
# Pass the two input components as `x` and `y` and the scalar output as `z`
# to [ZvsXY][gemseo.post.dataset.zvsxy.ZvsXY] to visualise the response surface.
