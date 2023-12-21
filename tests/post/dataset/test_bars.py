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
from __future__ import annotations

import pytest
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.bars import BarPlot
from gemseo.utils.testing.helpers import image_comparison


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """A dataset containing 5 samples of variables x1, x2 and x3."""
    dataset = Dataset()
    dataset.add_variable("x1", array([[-0.5], [3], [4], [6], [-2]]))
    dataset.add_variable("x2", array([[2], [2], [-1], [2], [5]]))
    dataset.add_variable("x3", array([[3], [1], [2], [3], [-0.5]]))
    dataset.index = [f"series_{i}" for i in range(5)]
    return dataset


TEST_PARAMETERS = {
    "default": ({}, {}, ["BarPlot"]),
    "xtick_rotation": (
        {},
        {"xtick_rotation": 45, "color": ["red", "blue", "yellow", "black", "green"]},
        ["BarPlot_xtick_rotation"],
    ),
}


@pytest.mark.parametrize(
    ("kwargs", "properties", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_bars_plot(tmp_path, kwargs, properties, dataset, baseline_images):
    """Test that bar plot generates the expected plot."""
    plot = BarPlot(dataset)
    for k, v in properties.items():
        setattr(plot, k, v)
    plot.execute(save=False)
