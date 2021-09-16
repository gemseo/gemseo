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

"""Test the class ZvsXY plotting a variable z versus two variables x and y."""

from __future__ import division, unicode_literals

import sys

import pytest
from matplotlib.testing.decorators import image_comparison
from numpy import array

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.zvsxy import ZvsXY
from gemseo.utils.py23_compat import PY2

pytestmark = pytest.mark.skipif(
    PY2, reason="image comparison does not work with python 2"
)


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


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to ParallelCoordinates._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default_z0": ({"x": "x", "y": "y", "z": "z", "properties": {}}, ["ZvsXY_z0"]),
    "default_z1": (
        {"x": "x", "y": "y", "z": "z", "z_comp": 1, "properties": {}},
        ["ZvsXY_z1"],
    ),
    "default_x0": ({"x": "z", "y": "y", "z": "z", "properties": {}}, ["ZvsXY_x0"]),
    "default_x1": (
        {"x": "z", "x_comp": 1, "y": "y", "z": "z", "properties": {}},
        ["ZvsXY_x1"],
    ),
    "default_y0": ({"x": "x", "y": "z", "z": "z", "properties": {}}, ["ZvsXY_y0"]),
    "default_y1": (
        {"x": "x", "y": "z", "y_comp": 1, "z": "z", "properties": {}},
        ["ZvsXY_y1"],
    ),
    "with_colormap": (
        {"x": "x", "y": "y", "z": "z", "properties": {"colormap": "viridis"}},
        ["ZvsXY_colormap"],
    ),
    "with_points": (
        {"x": "x", "y": "y", "z": "z", "add_points": True, "properties": {}},
        ["ZvsXY_points"],
    ),
    "with_points_and_color": (
        {
            "x": "x",
            "y": "y",
            "z": "z",
            "add_points": True,
            "properties": {"color": "white"},
        },
        ["ZvsXY_points_and_color"],
    ),
}


@pytest.mark.skipif(
    sys.version_info[:2] == (3, 6),
    reason="Image comparison based on Surfaces does not work with Python 3.6",
)
@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None, extensions=["png"])
def test_plot(kwargs, baseline_images, dataset, pyplot_close_all):
    """Test images created by ZvsXY._plot against references.

    Args:
        kwargs (dict): The optional arguments to pass to ZvsXY._plot.
        baseline_images (list): The images to be compared with.
        dataset (Dataset): A dataset.
        pyplot_close_all: Prevents figures aggregation.
    """
    ZvsXY(dataset)._plot(**kwargs)
