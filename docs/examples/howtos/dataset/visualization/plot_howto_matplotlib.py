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
"""# How to customise a plot with matplotlib

## Problem

The attributes of
[BaseDatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot]
cover the most common formatting needs,
but some fine-tuning (e.g. adding a grid, changing tick formatting)
requires direct access to the underlying matplotlib figure.

## Solution

Call `execute(save=False)` to retrieve the list of matplotlib `Figure` objects,
then use the standard matplotlib API to modify them before saving or displaying.

## Step-by-step guide
"""

from __future__ import annotations

from matplotlib import pyplot as plt

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.yvsx import YvsX
from gemseo.post.dataset.yvsx_settings import YvsX_Settings

# %%
# ### 1. Build the dataset
#
dataset = Dataset()
dataset.add_variable("a", [[1], [2], [3]])
dataset.add_variable("b", [[1], [0], [1]])

# %%
# Then,
# we define a [YvsX][gemseo.post.dataset.yvsx.YvsX] chart,
# which is a particular [BaseDatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot]:
yvsx = YvsX(dataset, YvsX_Settings(x="a", y="b"))

# %%
# and draw the figure.
figures = yvsx.execute(save=False)
figure = figures[0]

# %%
# ### 3. Customise the figure with matplotlib
#
# Access the `Axes` object and apply any standard matplotlib modifications:
ax = figure.axes[0]
ax.set_xlabel("A relevant x-label")
ax.set_ylabel("A relevant y-label")
ax.grid()

# %%
# ### 4. Save and display the figure
#
plt.savefig("foo.png")
plt.show()

# %%
# ## Summary
#
# Retrieve the matplotlib `Figure` objects from `execute(save=False)`,
# then modify them via `figure.axes[0]` before saving or displaying.
# Always check first whether a
# [BaseDatasetPlot][gemseo.post.dataset.base.BaseDatasetPlot]
# attribute (e.g. `xlabel`, `ylabel`) already covers the needed customisation;
# fall back to the matplotlib API only when it does not.
