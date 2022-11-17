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
"""Test the class Radar plotting samples using the radviz module from pandas."""
from __future__ import annotations

import pytest
from gemseo.post.dataset.radviz import Radar
from gemseo.problems.dataset.iris import IrisDataset
from gemseo.utils.testing import image_comparison
from matplotlib import pyplot as plt


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to Radar._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, {}, ["Radar"]),
    "with_properties": (
        {},
        {
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
        },
        ["Radar_properties"],
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
def test_plot(kwargs, properties, baseline_images, pyplot_close_all, fig_and_axes):
    """Test images created by Radar._plot against references."""
    dataset = IrisDataset()
    plot = Radar(dataset, classifier="specy")
    fig, axes = (
        (None, None) if not fig_and_axes else plt.subplots(figsize=plot.fig_size)
    )
    plot.execute(save=False, properties=properties, fig=fig, axes=axes)
