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
from gemseo.post.dataset.bars_settings import BarPlot_Settings


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
    "colormap": ({}, {"colormap": "viridis", "grid": False}, ["BarPlot_colormap"]),
    "properties": (
        {},
        {
            "xlabel": "foo",
            "title": "My Title",
            "xtick_rotation": 45.0,
            "color": ["red", "blue", "yellow", "black", "green"],
        },
        ["BarPlot_properties"],
    ),
    "no_annotation": (
        {"annotate": False},
        {},
        ["BarPlot_no_annotation"],
    ),
    "annotation_rotation": (
        {"annotation_rotation": 45.0},
        {},
        ["BarPlot_annotation_rotation"],
    ),
}


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [v[:2] for v in TEST_PARAMETERS.values()],
)
def test_bars_plot(kwargs, properties, dataset, snapshot_matplotlib) -> None:
    """Test that bar plot generates the expected plot."""
    settings = BarPlot_Settings(**kwargs, **properties)
    plot = BarPlot(dataset, settings)
    plot.execute(save=False)


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [v[:2] for v in TEST_PARAMETERS.values()],
    ids=TEST_PARAMETERS.keys(),
)
def test_bars_plotly(kwargs, properties, snapshot, dataset):
    """Test images created by BarPlot.execute against references for plotly."""
    pytest.importorskip("plotly")
    settings = BarPlot_Settings(**kwargs, **properties)
    plot = BarPlot(dataset, settings)
    figure = plot.execute(save=False, file_format="html")[0]
    assert figure.to_json() == snapshot
