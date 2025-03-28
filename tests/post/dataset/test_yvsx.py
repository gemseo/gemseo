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
"""Test the class YvsX plotting a variable y versus a variable x."""

from __future__ import annotations

import pytest
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.yvsx import YvsX
from gemseo.utils.testing.helpers import image_comparison


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 3 samples of variables x, y and z (dim(z)=2)."""
    sample1 = [0.0, 1.0, 1.0, 0.0]
    sample2 = [0.5, 0.0, 0.0, 1.0]
    sample3 = [1.0, 1.0, 1.0, 0.0]
    data_array = array([sample1, sample2, sample3])
    variable_names_to_n_components = {"x": 1, "y": 1, "z": 2}
    return Dataset.from_array(
        data_array,
        variable_names=["x", "y", "z"],
        variable_names_to_n_components=variable_names_to_n_components,
    )


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to YvsX._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({"x": "x", "y": "y"}, {}, ["YvsX"]),
    "with_color": (
        {"x": "x", "y": "y"},
        {"color": "red"},
        ["YvsX_color"],
    ),
    "with_style": (
        {"x": "x", "y": "y"},
        {"linestyle": "-"},
        ["YvsX_style"],
    ),
    "with_2d_output": ({"x": "x", "y": "z"}, {}, ["YvsX_2d_output"]),
    "with_2d_output_given_component": (
        {"x": "x", "y": ("z", 1)},
        {},
        ["YvsX_2d_output_given_component"],
    ),
    "with_2d_input": (
        {"x": "z", "y": "y"},
        {},
        ["YvsX_2d_input"],
    ),
    "with_2d_input_given_component": (
        {"x": ("z", 1), "y": "y"},
        {},
        ["YvsX_2d_input_given_component"],
    ),
    "with_properties": (
        {"x": "x", "y": "y"},
        {
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
            "grid": False,
        },
        ["YvsX_properties"],
    ),
}


@pytest.mark.parametrize(
    ("kwargs", "properties", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
@image_comparison(None)
def test_plot(kwargs, properties, baseline_images, dataset, fig_and_ax) -> None:
    """Test images created by YvsX._plot against references."""
    plot = YvsX(dataset, **kwargs)
    fig, ax = (None, None) if not fig_and_ax else plt.subplots(figsize=plot.fig_size)
    for k, v in properties.items():
        setattr(plot, k, v)
    plot.execute(save=False, fig=fig, ax=ax)
