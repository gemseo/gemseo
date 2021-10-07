# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import pytest
from matplotlib.testing.decorators import image_comparison
from numpy import array

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.radar_chart import RadarChart
from gemseo.utils.py23_compat import PY2

pytestmark = pytest.mark.skipif(
    PY2, reason="image comparison does not work with python 2"
)


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
    "default": ({}, ["RadarChart"]),
    "with_display_zero": ({"display_zero": False}, ["RadarChart_without_zero"]),
    "with_connect": ({"connect": True}, ["RadarChart_connect"]),
    "with_radial_ticks": ({"radial_ticks": True}, ["RadarChart_radial_ticks"]),
    "with_n_levels": ({"n_levels": 3}, ["RadarChart_n_levels"]),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None, extensions=["png"])
def test_plot(kwargs, baseline_images, dataset, pyplot_close_all):
    """Test images created by RadarChart._plot against references.

    Args:
        kwargs (dict): The optional arguments to pass to RadarChart._plot.
        baseline_images (list): The images to be compared with.
        dataset (Dataset): A dataset.
        pyplot_close_all: Prevents figures aggregation.
    """
    RadarChart(dataset)._plot(properties={}, **kwargs)
