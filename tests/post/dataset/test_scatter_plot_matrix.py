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
from __future__ import annotations

import re

import pytest
from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix
from gemseo.utils.testing import image_comparison
from matplotlib import pyplot as plt

from .test_andrews_curves import dataset  # noqa: F401


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to ScatterMatrix._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, {}, ["ScatterMatrix"]),
    "with_kde": ({"kde": True}, {}, ["ScatterMatrix_kde"]),
    "with_size": ({"size": 50}, {}, ["ScatterMatrix_size"]),
    "with_marker": ({"marker": "+"}, {}, ["ScatterMatrix_marker"]),
    "with_classifier": ({"classifier": "c"}, {}, ["ScatterMatrix_classifier"]),
    "with_names": ({"variable_names": ["x", "y"]}, {}, ["ScatterMatrix_names"]),
    "with_upper": ({"plot_lower": False}, {}, ["ScatterMatrix_lower"]),
    "with_lower": ({"plot_upper": False}, {}, ["ScatterMatrix_upper"]),
    "with_properties": (
        {},
        {"title": "The title"},
        ["ScatterMatrix_properties"],
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
    dataset,  # noqa: F811
    kwargs,
    properties,
    baseline_images,
    pyplot_close_all,
    fig_and_axes,
):
    """Test images created by ScatterMatrix._plot against references."""
    plot = ScatterMatrix(dataset, **kwargs)
    fig, axes = (
        (None, None) if not fig_and_axes else plt.subplots(figsize=plot.fig_size)
    )
    plot.execute(save=False, properties=properties, fig=fig, axes=axes)


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
