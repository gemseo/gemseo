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
"""Test the class Scatter plotting a variable y versus a variable x."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.scatter import Scatter
from gemseo.utils.testing import image_comparison
from matplotlib import pyplot as plt
from numpy import array


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 3 samples of variables x, y and z (dim(z)=2)."""
    dataset = Dataset()
    sample1 = [0.0, 1.0, 1.0, 0.0]
    sample2 = [0.5, 0.0, 0.0, 1.0]
    sample3 = [1.0, 1.0, 1.0, 0.0]
    data_array = array([sample1, sample2, sample3])
    sizes = {"x": 1, "y": 1, "z": 2}
    dataset.set_from_array(data_array, variables=["x", "y", "z"], sizes=sizes)
    return dataset


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to Scatter._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({"x": "x", "y": "y"}, {}, ["Scatter"]),
    "with_color": (
        {"x": "x", "y": "y"},
        {"color": "red"},
        ["Scatter_color"],
    ),
    "with_2d_output": ({"x": "x", "y": "z"}, {}, ["Scatter_2d_output"]),
    "with_2d_output_given_component": (
        {"x": "x", "y": "z", "y_comp": 1},
        {},
        ["Scatter_2d_output_given_component"],
    ),
    "with_2d_input": (
        {"x": "z", "y": "y"},
        {},
        ["Scatter_2d_input"],
    ),
    "with_2d_input_given_component": (
        {"x": "z", "y": "y", "x_comp": 1},
        {},
        ["Scatter_2d_input_given_component"],
    ),
    "with_properties": (
        {"x": "z", "y": "y"},
        {
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
        },
        ["Scatter_properties"],
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
    """Test images created by Scatter._plot against references."""
    plot = Scatter(dataset, **kwargs)
    fig, axes = (
        (None, None) if not fig_and_axes else plt.subplots(figsize=plot.fig_size)
    )
    plot.execute(save=False, properties=properties, fig=fig, axes=axes)
