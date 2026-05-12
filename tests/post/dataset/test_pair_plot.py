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
"""Test the class PairPlot plotting variables versus themselves."""

from __future__ import annotations

import pytest
from matplotlib import pyplot as plt
from scipy.interpolate import Rbf

from gemseo.post.dataset.pair_plot import PairPlot
from gemseo.post.dataset.pair_plot_settings import PairPlot_Settings
from gemseo.problems.dataset.iris import create_iris_dataset
from gemseo.utils.testing.helpers import assert_exception


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [
        ({}, {}),
        ({"use_kde": True}, {}),
        ({"size": 50}, {}),
        ({"marker": "+"}, {}),
        ({"classifier": "c"}, {}),
        ({"classifier": "c", "exclude_classifier": False}, {}),
        ({"variable_names": ["x", "y"]}, {}),
        ({"plot_lower": False}, {}),
        ({"plot_upper": False}, {}),
        ({"use_ranks": True}, {}),
        ({}, {"title": "The title", "grid": False}),
    ],
)
def test_plot(
    dataset,
    kwargs,
    properties,
    snapshot_matplotlib,
) -> None:
    """Test images created by PairPlot._plot against references."""
    settings = PairPlot_Settings(**kwargs, **properties)
    plot = PairPlot(dataset, settings)
    plot.execute(save=False)


@pytest.mark.parametrize(
    ("use_fig", "use_ax"), [(False, True), (True, False), (True, True)]
)
def test_fig_ax_error(dataset, use_fig, use_ax, snapshot) -> None:
    """Check the error raised when using fig and ax."""
    settings = PairPlot_Settings()
    fig, ax = plt.subplots(figsize=settings.fig_size)
    if not use_fig:
        fig = None
    if not use_ax:
        ax = None
    with assert_exception(ValueError, snapshot):
        PairPlot(dataset, settings).execute(save=False, fig=fig, ax=ax)


@pytest.mark.parametrize(
    "trend",
    ["linear", "quadratic", "cubic", "rbf", Rbf],
)
def test_trend(trend, quadratic_dataset, snapshot_matplotlib) -> None:
    """Check the use of a trend."""
    settings = PairPlot_Settings(trend=trend)
    PairPlot(quadratic_dataset, settings).execute(save=False)


def test_scatter_plot_option(dataset, snapshot_matplotlib) -> None:
    """Check the use of a scatter plot option."""
    settings = PairPlot_Settings(options={"alpha": 1.0})
    PairPlot(dataset, settings).execute(save=False)


def test_classifier_error(dataset, snapshot) -> None:
    """Check that an error is raised when the classifier is not variable name."""
    settings = PairPlot_Settings(classifier="wrong_name")
    with assert_exception(ValueError, snapshot):
        PairPlot(dataset, settings).execute(save=False)


def test_plot_lower_upper_error(dataset, snapshot) -> None:
    """Check that an error is raised when both plot_lower and plot_upper are False."""
    with assert_exception(ValueError, snapshot):
        PairPlot_Settings(plot_lower=False, plot_upper=False)


def test_2d_kde(snapshot_matplotlib):
    """Test the 2D KDE implementation for pair plots."""
    settings = PairPlot_Settings(use_scatter=False)
    dataset = create_iris_dataset()
    PairPlot(dataset, settings).execute(save=False)


def test_2d_kde_classifier(snapshot_matplotlib):
    """Test the 2D KDE implementation for pair plots with a classifier."""
    settings = PairPlot_Settings(use_scatter=False, classifier="specy")
    dataset = create_iris_dataset()
    PairPlot(dataset, settings).execute(save=False)


def test_iris(snapshot_matplotlib):
    """Test with the Iris dataset."""
    settings = PairPlot_Settings(use_kde=True, classifier="specy")
    dataset = create_iris_dataset()
    PairPlot(dataset, settings).execute(save=False)


def test_trend_surface(snapshot):
    """Test the error when using a trend with density surfaces."""
    with assert_exception(ValueError, snapshot):
        PairPlot_Settings(use_scatter=False, trend="linear")
