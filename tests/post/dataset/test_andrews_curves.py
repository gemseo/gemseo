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

from __future__ import annotations

import re

import pytest
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.andrews_curves import AndrewsCurves
from gemseo.utils.testing.helpers import image_comparison


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 4 samples of variables x, y and z and cluster c."""
    sample1 = [0.0, 0.0, 0.0, 1]
    sample2 = [1.0, 1.0, -1.0, 2]
    sample3 = [2.0, 2.0, -2.0, 2]
    sample4 = [3.0, 3.0, -3.0, 1]
    return Dataset.from_array(
        array([sample1, sample2, sample3, sample4]), ["x", "y", "z", "c"]
    )


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to ParallelCoordinates._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, ["AndrewsCurves"]),
    "with_properties": (
        {
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
            "grid": False,
        },
        ["AndrewsCurves_properties"],
    ),
}


@pytest.mark.parametrize(
    ("properties", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
@image_comparison(None)
def test_plot(properties, baseline_images, dataset, fig_and_ax) -> None:
    """Test images created by AndrewsCurves._plot against references."""
    plot = AndrewsCurves(dataset, classifier="c")
    fig, ax = (None, None) if not fig_and_ax else plt.subplots(figsize=plot.fig_size)
    for k, v in properties.items():
        setattr(plot, k, v)
    plot.execute(save=False, fig=fig, ax=ax)


def test_error(dataset) -> None:
    """Test an error is raised when a wrong name is given."""
    expected = "Classifier must be one of these names: c, x, y and z."
    with pytest.raises(ValueError, match=re.escape(expected)):
        AndrewsCurves(dataset, classifier="foo")._plot()
