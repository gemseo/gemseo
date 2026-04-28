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

import re

import pytest
from matplotlib import pyplot as plt
from scipy.interpolate import Rbf

from gemseo.post.dataset.pair_plot import PairPlot
from gemseo.post.dataset.pair_plot_settings import PairPlot_Settings
from gemseo.problems.dataset.iris import create_iris_dataset
from gemseo.utils.testing.helpers import image_comparison

# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to PairPlot._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, {}, ["PairPlot"]),
    "with_kde": ({"use_kde": True}, {}, ["PairPlot_kde"]),
    "with_size": ({"size": 50}, {}, ["PairPlot_size"]),
    "with_marker": ({"marker": "+"}, {}, ["PairPlot_marker"]),
    "with_classifier": ({"classifier": "c"}, {}, ["PairPlot_classifier"]),
    "with_classifier_and_exclude_classifier": (
        {"classifier": "c", "exclude_classifier": False},
        {},
        ["PairPlot_classifier_exclude"],
    ),
    "with_names": ({"variable_names": ["x", "y"]}, {}, ["PairPlot_names"]),
    "with_upper": ({"plot_lower": False}, {}, ["PairPlot_lower"]),
    "with_lower": ({"plot_upper": False}, {}, ["PairPlot_upper"]),
    "with_ranks": ({"use_ranks": True}, {}, ["PairPlot_ranks"]),
    "with_properties": (
        {},
        {"title": "The title", "grid": False},
        ["PairPlot_properties"],
    ),
}


@pytest.mark.parametrize(
    ("kwargs", "properties", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot(
    dataset,
    kwargs,
    properties,
    baseline_images,
) -> None:
    """Test images created by PairPlot._plot against references."""
    settings = PairPlot_Settings(**kwargs, **properties)
    plot = PairPlot(dataset, settings)
    plot.execute(save=False)


@pytest.mark.parametrize(
    ("use_fig", "use_ax"), [(False, True), (True, False), (True, True)]
)
def test_fig_ax_error(dataset, use_fig, use_ax) -> None:
    """Check the error raised when using fig and ax."""
    settings = PairPlot_Settings()
    fig, ax = plt.subplots(figsize=settings.fig_size)
    if not use_fig:
        fig = None
    if not use_ax:
        ax = None
    with pytest.raises(
        ValueError,
        match=re.escape("The arguments 'fig' and 'ax' are not supported by PairPlot."),
    ):
        PairPlot(dataset, settings).execute(save=False, fig=fig, ax=ax)


@pytest.mark.parametrize(
    ("trend", "baseline_images"),
    [
        ("linear", ["PairPlot_linear_trend"]),
        ("quadratic", ["PairPlot_quadratic_trend"]),
        ("cubic", ["PairPlot_cubic_trend"]),
        ("rbf", ["PairPlot_rbf_trend"]),
        (Rbf, ["PairPlot_custom_trend"]),
    ],
)
@image_comparison(None, tol=0.01)
def test_trend(trend, quadratic_dataset, baseline_images) -> None:
    """Check the use of a trend."""
    settings = PairPlot_Settings(trend=trend)
    PairPlot(quadratic_dataset, settings).execute(save=False)


@image_comparison(["PairPlot_scatter_plot_option"])
def test_scatter_plot_option(dataset) -> None:
    """Check the use of a scatter plot option."""
    settings = PairPlot_Settings(options={"alpha": 1.0})
    PairPlot(dataset, settings).execute(save=False)


def test_classifier_error(dataset) -> None:
    """Check that an error is raised when the classifier is not variable name."""
    settings = PairPlot_Settings(classifier="wrong_name")
    with pytest.raises(
        ValueError,
        match=re.escape(
            "wrong_name cannot be used as a classifier "
            "because it is not a variable name; "
            "available ones are: ['c', 'x', 'y', 'z']."
        ),
    ):
        PairPlot(dataset, settings).execute(save=False)


def test_plot_lower_upper_error(dataset) -> None:
    """Check that an error is raised when both plot_lower and plot_upper are False."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "At least one of the arguments 'plot_lower' and 'plot_upper' must be True."
        ),
    ):
        PairPlot_Settings(plot_lower=False, plot_upper=False)


@image_comparison(["PairPlot_2d_kde"])
def test_2d_kde():
    """Test the 2D KDE implementation for pair plots."""
    settings = PairPlot_Settings(use_scatter=False)
    dataset = create_iris_dataset()
    PairPlot(dataset, settings).execute(save=False)


@image_comparison(["PairPlot_2d_kde_classifier"])
def test_2d_kde_classifier():
    """Test the 2D KDE implementation for pair plots with a classifier."""
    settings = PairPlot_Settings(use_scatter=False, classifier="specy")
    dataset = create_iris_dataset()
    PairPlot(dataset, settings).execute(save=False)


@image_comparison(["PairPlot_iris"])
def test_iris():
    """Test with the Iris dataset."""
    settings = PairPlot_Settings(use_kde=True, classifier="specy")
    dataset = create_iris_dataset()
    PairPlot(dataset, settings).execute(save=False)


def test_trend_surface():
    """Test the error when using a trend with density surfaces."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The argument 'trend' must be 'none' "
            "when the argument 'use_scatter' is False."
        ),
    ):
        PairPlot_Settings(use_scatter=False, trend="linear")
