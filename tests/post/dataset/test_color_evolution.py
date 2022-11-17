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
"""Test the class ColorEvolution plotting samples on a colored timeline."""
from __future__ import annotations

import pytest
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.color_evolution import ColorEvolution
from gemseo.utils.testing import image_comparison
from matplotlib import pyplot as plt
from numpy import array


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 3 samples of variables x1, x2 and x3."""
    dataset = Dataset()
    dataset.add_variable("x1", array([[0], [1], [2]]))
    dataset.add_variable("x2", array([[0], [-1], [-2]]))
    dataset.add_variable("x3", array([[0.5, -0.5], [1.5, -1.5], [2.5, -2.5]]))
    return dataset


TEST_PARAMETERS = {
    "default": ({}, {}, ["ColorEvolution"]),
    "with_variables": ({"variables": ["x1", "x3"]}, {}, ["ColorEvolution_variables"]),
    "with_log": ({"use_log": True}, {}, ["ColorEvolution_log"]),
    "with_opacity": ({"opacity": 1.0}, {}, ["ColorEvolution_opacity"]),
    "with_properties": (
        {},
        {
            "colormap": "seismic",
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
        },
        ["ColorEvolution_properties"],
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
    """Test images created by ColorEvolution._plot against references."""
    plot = ColorEvolution(dataset, **kwargs)
    fig, axes = (
        (None, None) if not fig_and_axes else plt.subplots(figsize=plot.fig_size)
    )
    plot.execute(save=False, properties=properties, fig=fig, axes=axes)
