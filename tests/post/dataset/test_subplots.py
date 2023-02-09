# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test the creation of a plot with suplots."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.bars import BarPlot
from gemseo.post.dataset.yvsx import YvsX
from gemseo.utils.testing import image_comparison
from matplotlib import pyplot as plt
from numpy import array


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """The dataset to be plotted."""
    dataset = Dataset()
    dataset.add_variable("x1", array([[0.25, 0.35], [0.75, 0.85]]))
    dataset.add_variable("x2", array([[0.5], [0.25]]))
    dataset.add_variable("x3", array([[0.75], [0.25]]))
    dataset.row_names = ["series_1", "series_2"]
    return dataset


@image_comparison(["Subplots"])
def test_plot(dataset, pyplot_close_all):
    """Check the creation of a plot with subplots."""
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    plot_1 = BarPlot(dataset, n_digits=2)
    plot_1.colormap = "PiYG"
    plot_1.title = "Plot 1"
    plot_1.execute(save=False, fig=fig, axes=ax1)
    plot_2 = YvsX(dataset, "x2", "x3")
    plot_2.linestyle = "-d"
    plot_2.color = "r"
    plot_2.title = "Plot 2"
    plot_2.execute(save=False, fig=fig, axes=ax2)
    fig.suptitle("Plots 1 and 2")
