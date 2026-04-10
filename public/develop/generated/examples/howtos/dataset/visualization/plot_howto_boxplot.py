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
"""# Boxplot

## Problem

Visualise the distribution of one or more variables,
and compare distributions across multiple datasets.

## Solution

Use [Boxplot][gemseo.post.dataset.boxplot.Boxplot],
which renders one box per variable component
and supports centering, scaling, outlier removal,
confidence intervals, orientation, and overlay of additional datasets.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import hstack
from numpy import linspace

from gemseo.datasets.io_dataset import IODataset
from gemseo.post.dataset.boxplot import Boxplot
from gemseo.post.dataset.boxplot_settings import Boxplot_Settings

# %%
# ### 1. Build the datasets
#
inputs = linspace(-1, 1, 100)[:, None]

dataset = IODataset(dataset_name="Foo")
dataset.add_output_variable("y1", inputs**2)
dataset.add_output_variable("y2", hstack((inputs**3, inputs**4)))

other_dataset = IODataset(dataset_name="Bar")
other_dataset.add_output_variable("y1", -(inputs**2))
other_dataset.add_output_variable("y2", hstack((-(inputs**3), -(inputs**4))))

# %%
# ### 2. Plot a standard boxplot
#
plot = Boxplot(dataset)
plot.xlabel = "Variables"
plot.ylabel = "Values"
plot.title = "Standard boxplots"
plot.execute(save=False, show=True)

# %%
# ### 3. Center or scale the data
#
# Use `center=True` to subtract the mean from each variable:
plot = Boxplot(dataset, Boxplot_Settings(center=True))
plot.title = "With centering"
plot.execute(save=False, show=True)

# %%
# Use `scale=True` to normalise by the standard deviation:
plot = Boxplot(dataset, Boxplot_Settings(scale=True))
plot.title = "With scaling"
plot.execute(save=False, show=True)

# %%
# ### 4. Control outliers and confidence intervals
#
# Set `add_outliers=False` to hide individual outlier points:
plot = Boxplot(dataset, Boxplot_Settings(add_outliers=False))
plot.title = "Without outliers"
plot.execute(save=False, show=True)

# %%
# Set `add_confidence_interval=True` to display confidence intervals
# for the median:
plot = Boxplot(dataset, Boxplot_Settings(add_confidence_interval=True))
plot.title = "Confidence intervals"
plot.execute(save=False, show=True)

# %%
# ### 5. Change the orientation
#
# Set `use_vertical_bars=False` to display horizontal boxes:
plot = Boxplot(dataset, Boxplot_Settings(use_vertical_bars=False))
plot.title = "Horizontal bars"
plot.execute(save=False, show=True)

# %%
# ### 6. Overlay an additional dataset
#
# Pass extra datasets as positional arguments to compare distributions side by side:
plot = Boxplot(dataset, Boxplot_Settings(datasets=[other_dataset]))
plot.title = "Additional dataset"
plot.execute(save=False, show=True)

# %%
# ## Summary
#
# [Boxplot][gemseo.post.dataset.boxplot.Boxplot] displays the distribution
# of each variable component as a box.
# Use `center`, `scale`, `add_outliers`, `add_confidence_interval`
# and `use_vertical_bars` to adjust the representation,
# and pass additional datasets to compare distributions side by side.
