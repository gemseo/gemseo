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

from __future__ import annotations

import pytest
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.parallel_coordinates import ParallelCoordinates
from gemseo.post.dataset.parallel_coordinates_settings import (
    ParallelCoordinates_Settings,
)
from gemseo.utils.testing.helpers import assert_exception


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing two samples of the variables x1, x2 and x3."""
    sample1 = [0.0, 0.5, 1.0]
    sample2 = [0.2, 0.5, 0.8]
    sample3 = [1.0, 0.5, 0.0]
    return Dataset.from_array(
        array([sample1, sample2, sample3]), variable_names=["x1", "x2", "x3"]
    )


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [
        ({}, {}),
        ({"lower": 0.25}, {}),
        ({"upper": 0.75}, {}),
        ({"lower": 0.1, "upper": 0.75}, {}),
        (
            {},
            {
                "xlabel": "The xlabel",
                "ylabel": "The ylabel",
                "title": "The title",
                "grid": False,
            },
        ),
    ],
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
def test_plot(kwargs, properties, dataset, fig_and_ax, snapshot_matplotlib) -> None:
    """Test images created by ParallelCoordinates._plot against references."""
    settings = ParallelCoordinates_Settings(classifier="x1", **kwargs, **properties)
    plot = ParallelCoordinates(dataset, settings)
    fig, ax = (
        (None, None) if not fig_and_ax else plt.subplots(figsize=settings.fig_size)
    )
    plot.execute(save=False, fig=fig, ax=ax)


def test_wrong_classifier_name(dataset, snapshot) -> None:
    """Check that the message of the error raised when the classifier name is wrong."""
    with assert_exception(ValueError, snapshot):
        ParallelCoordinates(dataset, ParallelCoordinates_Settings(classifier="foo"))
