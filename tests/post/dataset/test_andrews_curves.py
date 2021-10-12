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

"""Test the class AndrewsCurves plotting samples as curves."""

from __future__ import division, unicode_literals

import pytest
from matplotlib.testing.decorators import image_comparison
from numpy import array

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.andrews_curves import AndrewsCurves
from gemseo.utils.py23_compat import PY2, PY3

pytestmark = [
    pytest.mark.skipif(PY2, reason="image comparison does not work with python 2"),
    pytest.mark.xfail(PY3, reason="fail with Python3 and coverage"),
]


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 4 samples of variables x, y and z and cluster c."""
    dataset = Dataset()
    sample1 = [0.0, 0.0, 0.0, 1]
    sample2 = [1.0, 1.0, -1.0, 2]
    sample3 = [2.0, 2.0, -2.0, 2]
    sample4 = [3.0, 3.0, -3.0, 1]
    dataset.set_from_array(
        array([sample1, sample2, sample3, sample4]), ["x", "y", "z", "c"]
    )
    return dataset


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to ParallelCoordinates._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, ["AndrewsCurves"]),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None, extensions=["png"])
def test_plot(kwargs, baseline_images, dataset, pyplot_close_all):
    """Test images created by AndrewsCurves._plot against references.

    Args:
        kwargs (dict): The optional arguments to pass to AndrewsCurves._plot.
        baseline_images (list): The images to be compared with.
        dataset (Dataset): A dataset.
        pyplot_close_all: Prevents figures aggregation.
    """
    AndrewsCurves(dataset)._plot({}, classifier="c")


def test_error(dataset):
    """Test an error is raised when a wrong name is given.

    Args:
        dataset (Dataset): A dataset.
    """
    expected = "Classifier must be one of these names: c, x, y, z"
    with pytest.raises(ValueError, match=expected):
        AndrewsCurves(dataset)._plot({}, classifier="foo")
