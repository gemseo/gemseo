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
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.utils.testing import image_comparison
from matplotlib import pyplot as plt
from numpy import array


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 2 samples of variables x1, x2 and x3."""
    dataset = Dataset()
    dataset.add_variable("x1", array([[-0.5], [3]]))
    dataset.add_variable("x2", array([[2], [2]]))
    dataset.add_variable("x3", array([[3], [1]]))
    dataset.row_names = ["series_1", "series_2"]
    return dataset


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to RadarChart._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, {}, ["RadarChart"]),
    "with_display_zero": ({"display_zero": False}, {}, ["RadarChart_without_zero"]),
    "with_connect": ({"connect": True}, {}, ["RadarChart_connect"]),
    "with_radial_ticks": ({"radial_ticks": True}, {}, ["RadarChart_radial_ticks"]),
    "with_n_levels": ({"n_levels": 3}, {}, ["RadarChart_n_levels"]),
    "with_properties": (
        {},
        {"title": "The title"},
        ["RadarChart_properties"],
    ),
}


@pytest.mark.parametrize(
    "kwargs, properties, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@pytest.mark.parametrize("fig_and_axes", [False, True])
@image_comparison(None)
def test_plot(
    kwargs, properties, baseline_images, dataset, pyplot_close_all, fig_and_axes
):
    """Test images created by RadarChart._plot against references."""
    plot = RadarChart(dataset, **kwargs)
    if fig_and_axes:
        fig = plt.figure(figsize=plot.fig_size)
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")
    else:
        fig, axes = (None, None)

    plot.execute(save=False, fig=fig, axes=axes, properties=properties)
