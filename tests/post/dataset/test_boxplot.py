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
from __future__ import annotations

import pytest
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.boxplot import Boxplot
from gemseo.utils.testing.helpers import image_comparison


@pytest.fixture(scope="module")
def dataset():
    """A dataset containing 3 samples of variables x, y and z (dim(z)=2)."""
    sample1 = [0.0, 1.0, 1.0, 0.0]
    sample2 = [0.75, 0.0, 0.5, 1.0]
    sample3 = [1.0, 1.0, 0.75, 0.0]
    data_array = array([sample1, sample2, sample3])
    variable_names_to_n_components = {"x": 1, "y": 1, "z": 2}
    dataset = Dataset.from_array(
        data_array,
        variable_names=["x", "y", "z"],
        variable_names_to_n_components=variable_names_to_n_components,
    )
    dataset.name = "A dataset"
    return dataset


@pytest.fixture(scope="module")
def other_dataset():
    """Another dataset containing 3 samples of variables x, y and z (dim(z)=2)."""
    sample1 = [0.0, 1.0, 1.0, 0.0]
    sample2 = [-0.75, -0.0, -0.5, -1.0]
    sample3 = [-1.0, -1.0, -0.75, -0.0]
    data_array = array([sample1, sample2, sample3])
    variable_names_to_n_components = {"x": 1, "y": 1, "z": 2}
    dataset = Dataset.from_array(
        data_array,
        variable_names=["x", "y", "z"],
        variable_names_to_n_components=variable_names_to_n_components,
    )
    dataset.name = "Another dataset"
    return dataset


TEST_PARAMETERS = {
    "default": ({}, {}, False, ["default"]),
    "variables": ({"variables": ["y", "z"]}, {}, False, ["variables"]),
    "scale": ({"scale": True}, {}, False, ["scale"]),
    "center": ({"center": True}, {}, False, ["center"]),
    "horizontal": ({"use_vertical_bars": False}, {}, False, ["horizontal"]),
    "confidence_interval": (
        {"add_confidence_interval": True},
        {},
        False,
        ["confidence_interval"],
    ),
    "outliers": ({"add_outliers": False}, {}, False, ["outliers"]),
    "option": ({"showmeans": True}, {}, False, ["option"]),
    "datasets": ({}, {}, True, ["datasets"]),
    "color": ({}, {"color": ["red", "blue"], "grid": False}, True, ["color"]),
}


@pytest.mark.parametrize(
    ("kwargs", "properties", "datasets", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
@image_comparison(None)
def test_plot(
    kwargs,
    properties,
    datasets,
    other_dataset,
    baseline_images,
    dataset,
    fig_and_ax,
) -> None:
    """Check Boxplot."""
    datasets = [other_dataset] if datasets else []
    plot = Boxplot(dataset, *datasets, **kwargs)
    fig, ax = (None, None) if not fig_and_ax else plt.subplots(figsize=plot.fig_size)
    for k, v in properties.items():
        setattr(plot, k, v)
    plot.execute(save=False, fig=fig, ax=ax)
