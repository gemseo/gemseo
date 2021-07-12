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

"""Test the class Radar plotting samples using the radviz module from pandas."""

from __future__ import division, unicode_literals

import pytest
from matplotlib.testing.decorators import image_comparison

from gemseo.post.dataset.radviz import Radar
from gemseo.problems.dataset.iris import IrisDataset
from gemseo.utils.py23_compat import PY2

pytestmark = pytest.mark.skipif(
    PY2, reason="image comparison does not work with python 2"
)


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to Radar._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, ["Radar"]),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None, extensions=["png"])
def test_plot(kwargs, baseline_images, pyplot_close_all):
    """Test images created by Radar._plot against references.

    Args:
        kwargs (dict): The optional arguments to pass to Radar._plot.
        baseline_images (list): The images to be compared with.
        pyplot_close_all: Prevents figures aggregation.
    """
    dataset = IrisDataset()
    Radar(dataset)._plot(properties={}, classifier="specy")
