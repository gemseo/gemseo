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

"""Test the class ScatterMatrix plotting variables versus themselves."""

from __future__ import division, unicode_literals

import re

import pytest
from matplotlib.testing.decorators import image_comparison

from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix
from gemseo.utils.py23_compat import PY2

from .test_andrews_curves import dataset  # noqa: F401

pytestmark = pytest.mark.skipif(
    PY2, reason="image comparison does not work with python 2"
)


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to ScatterMatrix._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, ["ScatterMatrix"]),
    "with_kde": ({"kde": True}, ["ScatterMatrix_kde"]),
    "with_size": ({"size": 50}, ["ScatterMatrix_size"]),
    "with_marker": ({"marker": "+"}, ["ScatterMatrix_marker"]),
    "with_classifier": ({"classifier": "c"}, ["ScatterMatrix_classifier"]),
    "with_names": ({"variable_names": ["x", "y"]}, ["ScatterMatrix_names"]),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None, extensions=["png"], tol=0.025)
def test_plot(dataset, kwargs, baseline_images, pyplot_close_all):  # noqa: F811
    """Test images created by ScatterMatrix._plot against references."""
    ScatterMatrix(dataset, **kwargs)._plot()


def test_plot_error(dataset):  # noqa: F811
    """Check that an error is raised when the classifier is not variable name."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "wrong_name cannot be used as a classifier "
            "because it is not a variable name; "
            "available ones are: ['c', 'x', 'y', 'z']."
        ),
    ):
        ScatterMatrix(dataset, classifier="wrong_name")._plot()
