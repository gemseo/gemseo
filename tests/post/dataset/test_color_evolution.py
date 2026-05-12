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
"""Test the class ColorEvolution plotting samples on a colored timeline."""

from __future__ import annotations

import pytest
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.color_evolution import ColorEvolution
from gemseo.post.dataset.color_evolution_settings import ColorEvolution_Settings


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 3 samples of variables x1, x2 and x3."""
    dataset = Dataset()
    dataset.add_variable("x1", array([[0], [1], [2]]))
    dataset.add_variable("x2", array([[0], [-1], [-2]]))
    dataset.add_variable("x3", array([[0.5, -0.5], [1.5, -1.5], [2.5, -2.5]]))
    return dataset


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [
        ({}, {}),
        ({"variables": ["x1", "x3"]}, {}),
        ({"use_log": True}, {}),
        ({"opacity": 1.0}, {}),
        (
            {},
            {
                "colormap": "seismic",
                "xlabel": "The xlabel",
                "ylabel": "The ylabel",
                "title": "The title",
            },
        ),
    ],
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
def test_plot(kwargs, properties, dataset, fig_and_ax, snapshot_matplotlib) -> None:
    """Test images created by ColorEvolution._plot against references."""
    settings = ColorEvolution_Settings(**kwargs, **properties)
    plot = ColorEvolution(dataset, settings)
    fig, ax = (
        (None, None) if not fig_and_ax else plt.subplots(figsize=settings.fig_size)
    )
    plot.execute(save=False, fig=fig, ax=ax)
