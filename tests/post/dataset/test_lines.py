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
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.lines import Lines
from gemseo.utils.testing import image_comparison
from matplotlib import pyplot as plt
from numpy import array


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 3 samples of variables x, y and z (dim(z)=2)."""
    dataset = Dataset()
    sample1 = [0.0, 1.0, 1.0, 0.0]
    sample2 = [0.75, 0.0, 0.5, 1.0]
    sample3 = [1.0, 1.0, 0.75, 0.0]
    data_array = array([sample1, sample2, sample3])
    sizes = {"x": 1, "y": 1, "z": 2}
    dataset.set_from_array(data_array, variables=["x", "y", "z"], sizes=sizes)
    return dataset


TEST_PARAMETERS = {
    "default": ({}, {}, ["Lines"]),
    "markers": ({"add_markers": True}, {}, ["Lines_markers"]),
    "variables": ({"variables": ["y"]}, {}, ["Lines_variables"]),
    "abscissa": (
        {"variables": ["y"], "abscissa_variable": "x"},
        {},
        ["Lines_abscissa"],
    ),
    "with_properties": (
        {"add_markers": True},
        {
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
            "legend_location": "upper right",
            "color": ["red", "black", "blue"],
            "linestyle": ["--", "-.", "-"],
            "marker": ["o", "v", "<"],
        },
        ["Lines_properties"],
    ),
    "with_other_properties": (
        {"add_markers": True},
        {"color": "red", "linestyle": "--", "marker": "v"},
        ["Lines_other_properties"],
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
    """Test images created by Lines.execute against references."""
    plot = Lines(dataset, **kwargs)
    fig, axes = (
        (None, None) if not fig_and_axes else plt.subplots(figsize=plot.fig_size)
    )
    plot.execute(save=False, fig=fig, axes=axes, properties=properties)
