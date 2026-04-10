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
"""# An output against an input

## Problem

Visualise how a variable `y` evolves as a function of another variable `x`.

## Solution

Use [YvsX][gemseo.post.dataset.yvsx.YvsX],
which renders a line or scatter plot of `y` as a function of `x`.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import linspace
from numpy import pi
from numpy import sin

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.yvsx import YvsX
from gemseo.post.dataset.yvsx_settings import YvsX_Settings

# %%
# ### 1. Build the dataset
#
inputs = linspace(0, 1, 10)[:, None]
outputs = sin(2 * pi * inputs)

dataset = Dataset()
dataset.add_variable("x", inputs, "inputs")
dataset.add_variable("y", outputs, "outputs")

# %%
# ### 2. Plot y vs x
#
plot = YvsX(dataset, YvsX_Settings(x="x", y="y"))
plot.linestyle = "--o"
plot.execute(save=False, show=True)

# %%
# ## Summary
#
# Pass the input variable name as `x` and the output variable name as `y`
# to [YvsX][gemseo.post.dataset.yvsx.YvsX] to visualise their relationship.
#
# ## One step further
#
# To plot several lines, please have a look on
# [Multiple variables as lines][multiple-variables-as-lines].
