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
"""# Bar chart

## Problem

Visualise and compare multiple series of values across several variables
as grouped bars.

## Solution

Use [BarPlot][gemseo.post.dataset.bars.BarPlot],
which renders one colour per dataset row (series)
with values grouped by variable name along the x-axis.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.bars import BarPlot
from gemseo.post.dataset.bars_settings import BarPlot_Settings

# %%
# ### 1. Build the dataset
#
# Each row is one series; columns are the variable components to compare:
dataset = Dataset()
dataset.add_variable("x1", array([[0.25, 0.35], [0.75, 0.85]]))
dataset.add_variable("x2", array([[0.5], [0.5]]))
dataset.add_variable("x3", array([[0.75], [0.25]]))
dataset.index = ["series_1", "series_2"]

# %%
# ### 2. Plot the bar chart
#
# Use `n_digits` to control the number of digits displayed on the bars
# and `colormap` to set the colour scheme:
plot = BarPlot(dataset, BarPlot_Settings(n_digits=2))
plot.colormap = "PiYG"
plot.execute(save=False, show=True)

# %%
# ## Summary
#
# [BarPlot][gemseo.post.dataset.bars.BarPlot] displays each dataset row as a series
# of grouped bars, one per variable component.
# The dataset index is used as series labels.
#
# ## One step further
#
# Set `file_format="html"` in `execute()` to produce an interactive plotly figure.
# See [How to generate an interactive plot][how-to-generate-an-interactive-plot] for details.
