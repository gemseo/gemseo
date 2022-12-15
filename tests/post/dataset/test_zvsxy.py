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
"""Test the class ZvsXY plotting a variable z versus two variables x and y."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.zvsxy import ZvsXY
from gemseo.utils.testing import image_comparison
from matplotlib import pyplot as plt
from numpy import array


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 5 samples of variables x, y and z (dim(z)=2)."""
    dataset = Dataset()
    sample1 = [0.0, 0.0, 1.0, 0.0]
    sample2 = [0.0, 1.0, 0.0, 1.0]
    sample3 = [1.0, 0.0, 0.0, 1.0]
    sample4 = [1.0, 1.0, 1.0, 0.0]
    sample5 = [0.5, 0.5, 0.5, 0.5]
    data_array = array([sample1, sample2, sample3, sample4, sample5])
    sizes = {"x": 1, "y": 1, "z": 2}
    dataset.set_from_array(data_array, variables=["x", "y", "z"], sizes=sizes)
    return dataset


other_dataset = Dataset()
sample1 = [0.0, 0.0, 1.0, 0.0]
sample2 = [0.0, 1.0, 0.0, 1.0]
sample3 = [1.0, 0.0, 0.0, 1.0]
sample4 = [1.0, 1.0, 1.0, 0.0]
sample5 = [0.5, 0.5, 0.5, 0.5]
data_array = array([[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.75, 0.75]])
sizes = {"x": 1, "y": 1, "z": 2}
other_dataset.set_from_array(data_array, variables=["x", "y", "z"], sizes=sizes)


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to ParallelCoordinates._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default_z0": ({"x": "x", "y": "y", "z": "z"}, {}, ["ZvsXY_z0"]),
    "default_z1": (
        {"x": "x", "y": "y", "z": "z", "z_comp": 1},
        {},
        ["ZvsXY_z1"],
    ),
    "default_x0": ({"x": "z", "y": "y", "z": "z"}, {}, ["ZvsXY_x0"]),
    "default_x1": (
        {"x": "z", "x_comp": 1, "y": "y", "z": "z"},
        {},
        ["ZvsXY_x1"],
    ),
    "default_y0": ({"x": "x", "y": "z", "z": "z"}, {}, ["ZvsXY_y0"]),
    "default_y1": (
        {"x": "x", "y": "z", "y_comp": 1, "z": "z"},
        {},
        ["ZvsXY_y1"],
    ),
    "with_colormap": (
        {"x": "x", "y": "y", "z": "z"},
        {"colormap": "viridis"},
        ["ZvsXY_colormap"],
    ),
    "with_points": (
        {"x": "x", "y": "y", "z": "z", "add_points": True},
        {},
        ["ZvsXY_points"],
    ),
    "with_points_and_color": (
        {
            "x": "x",
            "y": "y",
            "z": "z",
            "add_points": True,
        },
        {"color": "white"},
        ["ZvsXY_points_and_color"],
    ),
    "with_other_datasets": (
        {"x": "x", "y": "y", "z": "z", "other_datasets": [other_dataset]},
        {"color": "white"},
        ["ZvsXY_other_datasets"],
    ),
    "with_isolines": (
        {"x": "x", "y": "y", "z": "z", "fill": False},
        {},
        ["ZvsXY_isolines"],
    ),
    "with_levels": (
        {"x": "x", "y": "y", "z": "z", "levels": 2},
        {},
        ["ZvsXY_levels"],
    ),
    "with_properties": (
        {"x": "x", "y": "y", "z": "z"},
        {
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
        },
        ["ZvsXY_properties"],
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
    """Test images created by ZvsXY._plot against references."""
    plot = ZvsXY(dataset, **kwargs)
    fig, axes = (
        (None, None) if not fig_and_axes else plt.subplots(figsize=plot.fig_size)
    )
    plot.execute(save=False, fig=fig, axes=axes, properties=properties)
