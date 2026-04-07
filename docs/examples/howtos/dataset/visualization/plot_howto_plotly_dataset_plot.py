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
"""# How to generate an interactive plot

## Problem

GEMSEO uses [matplotlib](https://matplotlib.org/) by default for visualisation.
For web-based interactive figures, a plotly output is sometimes preferred.

## Solution

Pass `file_format="html"` to
[DatasetPlot.execute()][gemseo.post.dataset.base.BaseDatasetPlot.execute].
The method then returns a list of plotly figures
instead of saving or displaying a static matplotlib figure.
When `save=True` (default), the figures are saved to disk as HTML files.
When `show=True`, the figures are opened in the web browser.

!!! warning
    Some plots still do not support plotly.

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
dataset = Dataset()
dataset.add_variable("x1", array([[0.25, 0.35], [0.75, 0.85]]))
dataset.add_variable("x2", array([[0.5], [0.5]]))
dataset.add_variable("x3", array([[0.75], [0.25]]))
dataset.index = ["series_1", "series_2"]

# %%
# ### 2. Create the plot
#
plot = BarPlot(dataset, BarPlot_Settings(n_digits=2))
plot.colormap = "PiYG"

# %%
# ### 3. Execute with the HTML format
#
# Setting `file_format="html"` switches the rendering engine to plotly
# and returns a list of plotly figures:
plotly_figures = plot.execute(save=False, file_format="html")

# %%
# ### 4. Display the figure
#
plotly_figures[0]

# %%
# ## Summary
#
# Set `file_format="html"` in some
# [BaseDatasetPlot.execute()][gemseo.post.dataset.base.BaseDatasetPlot.execute] call
# to obtain an interactive plotly figure instead of a static matplotlib one.
#
# Some plots still cannot be generated with plotly.
#
# ## One step further
#
# Set `show=True` alongside `file_format="html"` to open the figure
# directly in a new tab of your web browser,
# which avoids any truncation that may occur in the web documentation.
