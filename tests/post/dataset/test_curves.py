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
"""Test the class Curves plotting samples of a 1D variable with curves."""

from __future__ import annotations

import pytest
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.curves import Curves
from gemseo.post.dataset.curves_settings import Curves_Settings


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing two samples of the 1D variable 'output'.

    This variable is plotted over the 1D mesh: [[0], [1]].

    The samples are [0., 1.] and [1., 0.].
    """
    sample1 = [0.0, 1.0]
    sample2 = [1.0, 0.0]
    data_array = array([sample1, sample2])
    dataset = Dataset.from_array(
        data_array,
        variable_names=["output"],
        variable_names_to_n_components={"output": 2},
    )
    dataset.misc["mesh"] = array([[0.0], [1.0]])
    return dataset


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [
        ({}, {}),
        ({"samples": [1]}, {}),
        (
            {},
            {
                "xlabel": "The xlabel",
                "ylabel": "The ylabel",
                "title": "The title",
                "color": ["red", "blue"],
                "linestyle": "-",
                "grid": False,
            },
        ),
    ],
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
def test_plot(kwargs, properties, dataset, fig_and_ax, snapshot_matplotlib) -> None:
    """Test images created by Curves._plot against references."""
    settings = Curves_Settings(mesh="mesh", variable="output", **kwargs, **properties)
    plot = Curves(dataset, settings)
    fig, ax = (
        (None, None) if not fig_and_ax else plt.subplots(figsize=settings.fig_size)
    )
    plot.execute(save=False, fig=fig, ax=ax)
