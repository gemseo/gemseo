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
"""Test the class ParallelCoordinates plotting samples with True/False lines."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.parallel_coordinates import ParallelCoordinates
from gemseo.utils.testing import image_comparison
from matplotlib import pyplot as plt
from numpy import array


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing two samples of the variables x1, x2 and x3."""
    dataset = Dataset()
    sample1 = [0.0, 0.5, 1.0]
    sample2 = [0.2, 0.5, 0.8]
    sample3 = [1.0, 0.5, 0.0]
    dataset.set_from_array(
        array([sample1, sample2, sample3]), variables=["x1", "x2", "x3"]
    )
    return dataset


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to ParallelCoordinates._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, {}, ["ParallelCoordinates"]),
    "with_lower": ({"lower": 0.25}, {}, ["ParallelCoordinates_lower"]),
    "with_upper": ({"upper": 0.75}, {}, ["ParallelCoordinates_upper"]),
    "with_lower_upper": (
        {"lower": 0.1, "upper": 0.75},
        {},
        ["ParallelCoordinates_lower_upper"],
    ),
    "with_properties": (
        {},
        {
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
        },
        ["ParallelCoordinates_properties"],
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
    """Test images created by ParallelCoordinates._plot against references."""
    plot = ParallelCoordinates(dataset, classifier="x1", **kwargs)
    fig, axes = (
        (None, None) if not fig_and_axes else plt.subplots(figsize=plot.fig_size)
    )
    plot.execute(save=False, fig=fig, axes=axes, properties=properties)
