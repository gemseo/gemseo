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
"""# Radar chart

## Problem

Compare multiple series of values across several variables on a single polar chart.

## Solution

Use [RadarChart][gemseo.post.dataset.radar_chart.RadarChart],
which renders one polygon per dataset row (series),
with each axis corresponding to a variable component.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.post.dataset.radar_chart_settings import RadarChart_Settings

# %%
# ### 1. Build the dataset
#
# Each row is one series; columns are the variable components to compare:
dataset = Dataset()
dataset.add_variable("x1", array([[0.2, 0.4, 0.5], [0.1, 0.3, 0.5]]))
dataset.add_variable("x2", array([[0.6], [0.5]]))
dataset.add_variable("x3", array([[0.8], [0.7]]))
dataset.index = ["series_1", "series_2"]

# %%
# ### 2. Plot the radar chart
#
# Set `connect=True` to close each polygon and `radial_ticks=True`
# to display tick labels along the radial axis.
# Use `rmin` and `rmax` to control the radial range:
plot = RadarChart(
    dataset, RadarChart_Settings(connect=True, radial_ticks=True, rmin=-0.5, rmax=1.0)
)
plot.execute(save=False, show=True)

# %%
# ## Summary
#
# [RadarChart][gemseo.post.dataset.radar_chart.RadarChart] draws one polygon per row
# of the dataset, making it easy to compare series across multiple variables at a glance.
