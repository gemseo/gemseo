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

import pytest
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.andrews_curves import AndrewsCurves
from gemseo.post.dataset.andrews_curves_settings import AndrewsCurves_Settings
from gemseo.utils.testing.helpers import assert_exception


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


@pytest.mark.parametrize(
    "properties",
    [
        {},
        {
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
            "grid": False,
        },
    ],
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
def test_plot(properties, dataset, fig_and_ax, snapshot_matplotlib) -> None:
    """Test images created by AndrewsCurves._plot against references."""
    settings = AndrewsCurves_Settings(classifier="c", **properties)
    plot = AndrewsCurves(dataset, settings)
    fig, ax = (
        (None, None) if not fig_and_ax else plt.subplots(figsize=settings.fig_size)
    )
    plot.execute(save=False, fig=fig, ax=ax)


def test_error(dataset, snapshot) -> None:
    """Test an error is raised when a wrong name is given."""
    with assert_exception(ValueError, snapshot):
        AndrewsCurves(dataset, AndrewsCurves_Settings(classifier="foo"))._plot()
