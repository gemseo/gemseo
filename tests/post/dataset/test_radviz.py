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
"""Test the class RadViz plotting samples using the radviz module from pandas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from matplotlib import pyplot as plt

from gemseo.post.dataset.radviz import RadViz
from gemseo.post.dataset.radviz_settings import RadViz_Settings
from gemseo.problems.dataset.iris import create_iris_dataset
from gemseo.utils.testing.helpers import assert_exception

if TYPE_CHECKING:
    from gemseo.datasets.dataset import Dataset


@pytest.fixture
def dataset() -> Dataset:
    """The Iris dataset."""
    return create_iris_dataset()


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [
        ({}, {}),
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
def test_plot(dataset, kwargs, properties, fig_and_ax, snapshot_matplotlib) -> None:
    """Test images created by RadViz._plot against references."""
    settings = RadViz_Settings(classifier="specy", **properties)
    plot = RadViz(dataset, settings)
    fig, ax = (
        (None, None) if not fig_and_ax else plt.subplots(figsize=settings.fig_size)
    )
    plot.execute(save=False, fig=fig, ax=ax)


def test_classifier_error(dataset, snapshot) -> None:
    """Check the error returned when setting a classifier that is not a variable."""
    with assert_exception(ValueError, snapshot):
        RadViz(dataset, RadViz_Settings(classifier="foo")).execute()
