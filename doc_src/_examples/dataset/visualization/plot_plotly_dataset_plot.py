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
"""
Interactive visualization using plotly
======================================

By default,
|g| uses `matplotlib <https://matplotlib.org/>`__ for data visualization.
However,
for web-based interactive visualizations,
`plotly <https://plotly.com/python/>`__ can be appreciated.
For this reason,
|g| proposes plotly versions of some visualizations
which can be generated using the option ``file_format="html"``
of the method :meth:`.DatasetPlot.execute`.
In that case,
this method returns a list of plotly figures.
When ``save=True`` (default), the figures are saved on the disk.
When ``show=True``, the figures are displayed in the web browser.
"""

from __future__ import annotations

from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.bars import BarPlot

# %%
# In this example,
# we create a simple dataset:
dataset = Dataset()
dataset.add_variable("x1", array([[0.25, 0.35], [0.75, 0.85]]))
dataset.add_variable("x2", array([[0.5], [0.5]]))
dataset.add_variable("x3", array([[0.75], [0.25]]))
dataset.index = ["series_1", "series_2"]

# %%
# then,
# we create a :class:`.BarPlot`:
plot = BarPlot(dataset, n_digits=2)
plot.colormap = "PiYG"

# %%
# generate the plotly figure:
plotly_figure = plot.execute(save=False, file_format="html")[0]

# %%
# and visualize it:
plotly_figure

# %%
# .. warning::
#
#    The plotly figures can be truncated both at the bottom and on the right
#    in the web documentation.
#    This an issue related to the documentation plugin. The problem is
#    solved by resizing the window.
#    Alternatively, setting ``show=True`` will display the figure in a
#    new tab of your web browser.
