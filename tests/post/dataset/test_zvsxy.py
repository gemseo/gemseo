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
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.zvsxy import ZvsXY
from gemseo.post.dataset.zvsxy_settings import ZvsXY_Settings


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 5 samples of variables x, y and z (dim(z)=2)."""
    sample1 = [0.0, 0.0, 1.0, 0.0]
    sample2 = [0.0, 1.0, 0.0, 1.0]
    sample3 = [1.0, 0.0, 0.0, 1.0]
    sample4 = [1.0, 1.0, 1.0, 0.0]
    sample5 = [0.5, 0.5, 0.5, 0.5]
    data_array = array([sample1, sample2, sample3, sample4, sample5])
    variable_name_to_n_components = {"x": 1, "y": 1, "z": 2}
    return Dataset.from_array(
        data_array,
        variable_names=["x", "y", "z"],
        variable_name_to_n_components=variable_name_to_n_components,
    )


sample1 = [0.0, 0.0, 1.0, 0.0]
sample2 = [0.0, 1.0, 0.0, 1.0]
sample3 = [1.0, 0.0, 0.0, 1.0]
sample4 = [1.0, 1.0, 1.0, 0.0]
sample5 = [0.5, 0.5, 0.5, 0.5]
data_array = array([[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.75, 0.75]])
variable_name_to_n_components = {"x": 1, "y": 1, "z": 2}
other_dataset = Dataset.from_array(
    data_array,
    variable_names=["x", "y", "z"],
    variable_name_to_n_components=variable_name_to_n_components,
)
other_dataset.name = "foo"


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [
        ({"x": "x", "y": "y", "z": "z"}, {}),
        ({"x": "x", "y": "y", "z": ("z", 1)}, {}),
        ({"x": "z", "y": "y", "z": "z"}, {}),
        ({"x": ("z", 1), "y": "y", "z": "z"}, {}),
        ({"x": "x", "y": "z", "z": "z"}, {}),
        ({"x": "x", "y": ("z", 1), "z": "z"}, {}),
        ({"x": "x", "y": "y", "z": "z"}, {"colormap": "viridis"}),
        ({"x": "x", "y": "y", "z": "z", "add_points": True}, {}),
        (
            {"x": "x", "y": "y", "z": "z", "add_points": True},
            {"color": "white"},
        ),
        (
            {"x": "x", "y": "y", "z": "z", "other_datasets": [other_dataset]},
            {"color": "white", "grid": True},
        ),
        ({"x": "x", "y": "y", "z": "z", "fill": False}, {}),
        ({"x": "x", "y": "y", "z": "z", "levels": 2}, {}),
        (
            {"x": "x", "y": "y", "z": "z"},
            {
                "xlabel": "The xlabel",
                "ylabel": "The ylabel",
                "zlabel": "The zlabel",
                "title": "The title",
            },
        ),
    ],
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
def test_plot(kwargs, properties, dataset, fig_and_ax, snapshot_matplotlib) -> None:
    """Test images created by ZvsXY._plot against references."""
    settings = ZvsXY_Settings(**kwargs, **properties)
    plot = ZvsXY(dataset, settings)
    fig, ax = (
        (None, None) if not fig_and_ax else plt.subplots(figsize=settings.fig_size)
    )
    plot.execute(save=False, fig=fig, ax=ax)
