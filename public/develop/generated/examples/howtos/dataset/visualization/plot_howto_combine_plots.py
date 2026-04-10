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
"""# Combine different plots into one

## Problem

You want to display several dataset plots side by side in a single figure.

## Solution

Create a matplotlib `Figure` with the desired layout,
then pass the `fig` and the target `Axes` object to each plot's `execute()` call.
This lets every [BaseDatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot]
draw itself into the subplot you choose.

## Step-by-step guide
"""

from __future__ import annotations

from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.bars import BarPlot
from gemseo.post.dataset.bars_settings import BarPlot_Settings
from gemseo.post.dataset.yvsx import YvsX
from gemseo.post.dataset.yvsx_settings import YvsX_Settings

# %%
# ### 1. Build the dataset
#
# Each row is one series; `x1` has two components while `x2` and `x3` are scalars:
dataset = Dataset()
dataset.add_variable("x1", array([[0.25, 0.35], [0.75, 0.85]]))
dataset.add_variable("x2", array([[0.5], [0.25]]))
dataset.add_variable("x3", array([[0.75], [0.25]]))
dataset.index = ["series_1", "series_2"]

# %%
# ### 2. Create the shared figure
#
# Use `plt.subplots` to create a figure with as many `Axes` as plots needed:
fig, (ax1, ax2) = plt.subplots(ncols=2)

# %%
# ### 3. Draw the first plot into the left subplot
#
# Pass `fig` and `ax1` to `execute()` so the bar chart renders in the left panel.
# Set `save=False` to prevent each plot from creating its own figure:
plot_1 = BarPlot(dataset, BarPlot_Settings(n_digits=2))
plot_1.colormap = "PiYG"
plot_1.title = "Plot 1"
plot_1.execute(save=False, fig=fig, ax=ax1)

# %%
# ### 4. Draw the second plot into the right subplot
#
# Pass `fig` and `ax2` to `execute()` so the scatter renders in the right panel:
plot_2 = YvsX(dataset, YvsX_Settings(x="x2", y="x3"))
plot_2.linestyle = "-d"
plot_2.color = "r"
plot_2.title = "Plot 2"
plot_2.execute(save=False, fig=fig, ax=ax2)

# %%
# ### 5. Add a shared title and display
#
fig.suptitle("Plots 1 and 2")
plt.show()

# %%
# ## Summary
#
# To combine multiple dataset plots into one figure:
#
# 1. Create a matplotlib `Figure` with `plt.subplots`.
# 2. Instantiate each plot and configure its attributes.
# 3. Call `execute(save=False, fig=fig, ax=ax)` for each plot,
#    passing the shared figure and the target `Axes` object.
#
# ## One step further
#
# For further fine-tuning of the combined figure,
# use the standard matplotlib API directly on the returned `Axes` objects.
# See [How to customise a plot with matplotlib][how-to-customize-a-plot-with-matplotlib]
# for details.
