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
"""Test the class Heatmap, either rectangular or symmetric."""

from __future__ import annotations

import json

import pytest
from matplotlib import pyplot as plt
from numpy import array
from numpy import nan

from gemseo.dataset.dataset import Dataset
from gemseo.post.dataset.heatmap import Heatmap
from gemseo.post.dataset.heatmap_settings import Heatmap_Settings
from gemseo.post.dataset.plot._heatmap_utils import compute_centered_bounds
from gemseo.util.testing.helper import assert_exception


@pytest.fixture(scope="module")
def rectangular_dataset() -> Dataset:
    """A dataset containing 3 samples of variables x1, x2 and x3."""
    dataset = Dataset()
    dataset.add_variable("x1", array([[0], [1], [2]]))
    dataset.add_variable("x2", array([[0], [-1], [-2]]))
    dataset.add_variable("x3", array([[0.5, -0.5], [nan, -1.5], [2.5, -2.5]]))
    return dataset


@pytest.fixture(scope="module")
def symmetric_dataset() -> Dataset:
    """A dataset containing a symmetric 4x4 matrix over x1, x2 and x3 (2 components).

    This dataset includes a nan.
    """
    dataset = Dataset()
    dataset.add_variable("x1", array([[0.0], [0.5], [-0.3], [0.1]]))
    dataset.add_variable("x2", array([[0.5], [0.0], [nan], [-0.4]]))
    dataset.add_variable(
        "x3", array([[-0.3, 0.1], [nan, -0.4], [0.0, 0.6], [0.6, 0.0]])
    )
    return dataset


RECTANGULAR_SETTINGS = [
    Heatmap_Settings(),
    Heatmap_Settings(variables=("x1", "x3")),
    Heatmap_Settings(
        use_log=True,
        opacity=1.0,
        colormap="seismic",
        xlabel="The xlabel",
        ylabel="The ylabel",
        title="The title",
    ),
]
RECTANGULAR_SETTINGS_IDS = ["default", "variables", "use_log_opacity_and_properties"]

SYMMETRIC_SETTINGS = [
    Heatmap_Settings(symmetric=True),
    Heatmap_Settings(symmetric=True, variables=("x1", "x3")),
    Heatmap_Settings(
        symmetric=True,
        # Keep center at its meaningful default (0.0): this pins use_log taking
        # precedence over centering, rather than trivially disabling it via None.
        annotate=True,
        use_log=True,
        opacity=1.0,
        colormap="seismic",
        xlabel="The xlabel",
        ylabel="The ylabel",
        title="The title",
    ),
]
SYMMETRIC_SETTINGS_IDS = ["default", "variables", "use_log_precedence_properties"]


@pytest.mark.parametrize("settings", RECTANGULAR_SETTINGS, ids=RECTANGULAR_SETTINGS_IDS)
def test_plot_rectangular(settings, rectangular_dataset, snapshot_matplotlib) -> None:
    """Test images created by Heatmap._plot with symmetric=False against references."""
    plot = Heatmap(rectangular_dataset, settings)
    plot.execute(save=False)


@pytest.mark.parametrize("settings", SYMMETRIC_SETTINGS, ids=SYMMETRIC_SETTINGS_IDS)
def test_plot_symmetric(settings, symmetric_dataset, snapshot_matplotlib) -> None:
    """Test images created by Heatmap._plot with symmetric=True against references."""
    plot = Heatmap(symmetric_dataset, settings)
    plot.execute(save=False)


@pytest.mark.parametrize("symmetric", [False, True])
def test_plot_given_fig_and_ax(
    rectangular_dataset, symmetric_dataset, symmetric
) -> None:
    """Check that execute() reuses a pre-built fig and ax instead of new ones."""
    dataset = symmetric_dataset if symmetric else rectangular_dataset
    settings = Heatmap_Settings(symmetric=symmetric)
    plot = Heatmap(dataset, settings)
    fig, ax = plt.subplots(figsize=settings.fig_size)
    plot.execute(save=False, fig=fig, ax=ax)
    assert plot.figures == [fig]
    assert ax in fig.axes
    assert ax.images


def test_non_square_dataset_error(snapshot) -> None:
    """Check that an error is raised when the dataset is not square."""
    dataset = Dataset.from_array(array([[0.0, 0.5], [0.5, 0.0], [0.1, 0.2]]))
    with assert_exception(ValueError, snapshot):
        Heatmap(dataset, Heatmap_Settings(symmetric=True))


@pytest.mark.parametrize(
    ("data", "center", "expected"),
    [
        (array([1.0, 2.0]), 0.0, None),
        (array([-1.0, 2.0]), 0.0, (-1.0, 2.0)),
    ],
)
def test_compute_centered_bounds(data, center, expected) -> None:
    """Check compute_centered_bounds: None, non-straddling and straddling cases."""
    assert compute_centered_bounds(data, center) == expected


PLOTLY_RECTANGULAR_SETTINGS = [
    Heatmap_Settings(),
    Heatmap_Settings(variables=("x1", "x3")),
    Heatmap_Settings(
        use_log=True,
        opacity=1.0,
        annotate=True,
        colormap="rdbu",
        xlabel="The xlabel",
        ylabel="The ylabel",
        title="The title",
    ),
]

PLOTLY_SYMMETRIC_SETTINGS = [
    Heatmap_Settings(symmetric=True),
    Heatmap_Settings(symmetric=True, variables=("x1", "x3")),
    Heatmap_Settings(
        symmetric=True,
        # Keep center at its meaningful default (0.0): this pins use_log taking
        # precedence over centering, rather than trivially disabling it via None.
        annotate=True,
        use_log=True,
        opacity=1.0,
        colormap="rdbu",
        xlabel="The xlabel",
        ylabel="The ylabel",
        title="The title",
    ),
]


@pytest.mark.parametrize(
    "settings", PLOTLY_RECTANGULAR_SETTINGS, ids=RECTANGULAR_SETTINGS_IDS
)
def test_plot_plotly_rectangular(
    settings, rectangular_dataset, snapshot_allclose
) -> None:
    """Test the figure created by Heatmap.execute with symmetric=False."""
    pytest.importorskip("plotly")
    plot = Heatmap(rectangular_dataset, settings)
    figure = plot.execute(save=False, file_format="html")[0]
    assert json.loads(figure.to_json()) == snapshot_allclose(rtol=1e-2)


@pytest.mark.parametrize(
    "settings", PLOTLY_SYMMETRIC_SETTINGS, ids=SYMMETRIC_SETTINGS_IDS
)
def test_plot_plotly_symmetric(settings, symmetric_dataset, snapshot_allclose) -> None:
    """Test the figure created by Heatmap.execute with symmetric=True."""
    pytest.importorskip("plotly")
    plot = Heatmap(symmetric_dataset, settings)
    figure = plot.execute(save=False, file_format="html")[0]
    assert json.loads(figure.to_json()) == snapshot_allclose(rtol=1e-2)
