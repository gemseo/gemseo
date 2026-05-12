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
"""Test the class RadarChart plotting samples on a radar chart."""

from __future__ import annotations

import pytest
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.post.dataset.radar_chart_settings import RadarChart_Settings


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 2 samples of variables x1, x2 and x3."""
    dataset = Dataset()
    dataset.add_variable("x1", array([[-0.5], [3]]))
    dataset.add_variable("x2", array([[2], [2]]))
    dataset.add_variable("x3", array([[3], [1]]))
    dataset.index = ["series_1", "series_2"]
    return dataset


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [
        ({}, {}),
        ({"display_zero": False}, {}),
        ({"connect": True}, {}),
        ({"radial_ticks": True}, {}),
        ({"n_levels": 3}, {}),
        (
            {},
            {
                "title": "The title",
                "linestyle": ["-", "--"],
                "rmin": -1,
                "rmax": 4,
                "color": "red",
                "grid": False,
            },
        ),
        ({"scientific_notation": False}, {"colormap": "viridis"}),
    ],
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
def test_plot(kwargs, properties, dataset, fig_and_ax, snapshot_matplotlib) -> None:
    """Test images created by RadarChart._plot against references."""
    settings = RadarChart_Settings(**kwargs, **properties)
    plot = RadarChart(dataset, settings)
    if fig_and_ax:
        fig = plt.figure(figsize=settings.fig_size)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")
    else:
        fig, ax = (None, None)
    plot.execute(save=False, fig=fig, ax=ax)
