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

"""Test the class ParallelCoordinates plotting samples with True/False lines."""

from __future__ import division, unicode_literals

import pytest
from matplotlib.testing.decorators import image_comparison
from numpy import array

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.parallel_coordinates import ParallelCoordinates
from gemseo.utils.py23_compat import PY2

pytestmark = pytest.mark.skipif(
    PY2, reason="image comparison does not work with python 2"
)


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
    "default": ({}, ["ParallelCoordinates"]),
    "with_lower": ({"lower": 0.25}, ["ParallelCoordinates_lower"]),
    "with_upper": ({"upper": 0.75}, ["ParallelCoordinates_upper"]),
    "with_lower_upper": (
        {"lower": 0.1, "upper": 0.75},
        ["ParallelCoordinates_lower_upper"],
    ),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None, extensions=["png"])
def test_plot(kwargs, baseline_images, dataset, pyplot_close_all):
    """Test images created by ParallelCoordinates._plot against references.

    Args:
        kwargs (dict): The optional arguments to pass to ParallelCoordinates._plot.
        baseline_images (list): The images to be compared with.
        dataset (Dataset): A dataset.
        pyplot_close_all: Prevents figures aggregation.
    """
    ParallelCoordinates(dataset)._plot(properties={}, classifier="x1", **kwargs)
