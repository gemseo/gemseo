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
"""# Multiple variables as lines

## Problem

Visualise the evolution of several variables against the dataset index
on a single chart.

## Solution

Use [Lines][gemseo.post.dataset.lines.Lines],
which renders one line per selected variable,
with customisable line styles.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import cos
from numpy import linspace
from numpy import pi
from numpy import sin

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.lines import Lines
from gemseo.post.dataset.lines_settings import Lines_Settings

# %%
# ### 1. Build the dataset
#
inputs = linspace(0, 1, 10)[:, None]
outputs_1 = sin(2 * pi * inputs)
outputs_2 = cos(2 * pi * inputs)

dataset = Dataset()
dataset.add_variable("x", inputs, "inputs")
dataset.add_variable("y1", outputs_1, "outputs")
dataset.add_variable("y2", outputs_2, "outputs")

# %%
# ### 2. Plot the lines
#
# Pass the variable names to plot and assign one line style per variable:
plot = Lines(dataset, Lines_Settings(variables=("y1", "y2")))
plot.linestyle = ["--", "-"]
plot.execute(save=False, show=True)

# %%
# ## Summary
#
# [Lines][gemseo.post.dataset.lines.Lines] plots several variables
# against the dataset index on the same axes.
# Use `variables` to select which ones to display
# and `linestyle` to distinguish them visually.
#
# ## One step further
#
# Retrieving the figure as explained in
# [How to customise a plot with matplotlib][how-to-customize-a-plot-with-matplotlib]
# may help to add other customized lines in this plot.
#
# Set `file_format="html"` in `execute()` to produce an interactive plotly figure.
# See [How to generate an interactive plot][how-to-generate-an-interactive-plot] for details.
