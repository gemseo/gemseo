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

from dataclasses import fields
from pathlib import Path
from unittest import mock

import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset._matplotlib import plot
from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot
from gemseo.post.dataset._plotly.lines import Figure as PlotlyFigure
from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.post.dataset.lines import Lines
from gemseo.post.dataset.yvsx import YvsX
from gemseo.utils.testing.helpers import concretize_classes


@pytest.fixture()
def yvsx() -> YvsX:
    """A plot of type YvsX."""
    dataset = Dataset()
    dataset.add_variable("x", array([[1], [2]]))
    dataset.add_variable("y", array([[1], [2]]))
    return YvsX(dataset, x="x", y="y")


def test_empty_dataset():
    dataset = Dataset()
    with pytest.raises(ValueError):
        YvsX(dataset, x="x", y="y")


def test_get_label():
    dataset = Dataset.from_array(
        array([[1, 2]]), variable_names=["x"], variable_names_to_n_components={"x": 2}
    )
    with concretize_classes(DatasetPlot):
        post = DatasetPlot(dataset)
    label, varname = post._get_label(("foo", "x", 0))
    assert label == "x[0]"
    assert varname == ("foo", "x", 0)

    label, varname = post._get_label("x")
    assert label == "x"
    assert varname == ("parameters", "x", 0)


@pytest.fixture()
def dataset() -> Dataset:
    """A very simple dataset with a single value: x=[1]."""
    return Dataset.from_array(
        array([[1]]), variable_names=["x"], variable_names_to_n_components={"x": 1}
    )


@pytest.fixture()
def dataset_plot(dataset):
    """A simple dataset plot from a dataset with a single value: x=[1]."""
    with concretize_classes(DatasetPlot):
        return DatasetPlot(dataset)


@pytest.mark.parametrize(
    ("attribute", "default_value"),
    [
        ("xlabel", ""),
        ("ylabel", ""),
        ("zlabel", ""),
        ("title", ""),
        ("font_size", 10),
        ("labels", {}),
    ],
)
def test_property_default_values(dataset_plot, attribute, default_value):
    """Check the default values of properties."""
    assert getattr(dataset_plot, attribute) == default_value


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("xlabel", "dummy_xlabel"),
        ("ylabel", "dummy_ylabel"),
        ("zlabel", "dummy_zlabel"),
        ("title", "dummy_title"),
        ("font_size", 2),
        ("labels", {"x": "dummy_label"}),
    ],
)
def test_setters(dataset_plot, attribute, value):
    """Check the attribute setters."""
    setattr(dataset_plot, attribute, value)
    assert getattr(dataset_plot, attribute) == value


def test_get_figure_and_axes_from_existing_fig_and_axes(dataset_plot, dataset):
    """Check that get_figure_and_axes using fig and axes returns fig and axes."""
    fig, axes = plt.subplots()
    plot = Lines(dataset)
    plot.xlabel = "foo"
    figures = plot.execute(show=False, save=False, fig=fig, axes=axes)
    assert figures == [fig]
    assert axes.get_figure() == fig
    assert axes.get_xlabel() == "foo"


def test_get_figure_and_axes_from_scratch(dataset, dataset_plot):
    """Check that get_figure_and_axes without fig and axes returns new fig and axes."""
    with concretize_classes(MatplotlibPlot):
        plot = MatplotlibPlot(dataset, dataset_plot._common_settings, (), None, None)
    fig, axes = plot._get_figure_and_axes(None, None)
    assert isinstance(fig, Figure)
    assert isinstance(axes, Axes)


def test_get_figure_and_axes_from_axes_only(dataset, dataset_plot):
    """Check that get_figure_and_axes with axes and without fig raises a ValueError."""
    _, axes = plt.subplots()
    with concretize_classes(MatplotlibPlot):
        plot = MatplotlibPlot(dataset, dataset_plot._common_settings, (), None, axes)
    with pytest.raises(
        ValueError, match="The figure associated with the given axes is missing."
    ):
        plot._get_figure_and_axes(None, axes)


def test_get_figure_and_axes_from_figure_only(dataset, dataset_plot):
    """Check that get_figure_and_axes without axes and with fig raises a ValueError."""
    fig, _ = plt.subplots()
    with concretize_classes(MatplotlibPlot):
        plot = MatplotlibPlot(dataset, dataset_plot._common_settings, (), fig, None)
    with pytest.raises(
        ValueError, match="The axes associated with the given figure are missing."
    ):
        plot._get_figure_and_axes(fig, None)


@pytest.mark.parametrize(
    ("kwargs", "relative", "expected"),
    [
        ({}, False, "yvs_x.png"),
        ({"file_path": "foo.png"}, True, "foo.png"),
        ({"file_name": "bar"}, False, "bar.png"),
        ({"directory_path": "foo"}, True, Path("foo") / "yvs_x.png"),
        ({"file_path": Path("foo") / "bar.png"}, True, Path("foo") / "bar.png"),
    ],
)
def test_save(yvsx, tmp_wd, kwargs, expected, relative):
    """Check that saving works correctly."""
    if "directory_path" in kwargs:
        Path(kwargs["directory_path"]).mkdir()

    if "file_path" in kwargs and Path(kwargs["file_path"]).parent != Path("."):
        Path(kwargs["file_path"]).parent.mkdir()

    yvsx.execute(save=True, **kwargs)
    file_path = Path(expected) if relative else tmp_wd / expected
    assert file_path.exists()
    assert yvsx.output_files == [str(file_path)]


def test_plot_settings(yvsx):
    """Check that properties and setters read and write plot settings."""
    for field in fields(yvsx._common_settings):
        name = field.name
        assert getattr(yvsx, name) == getattr(yvsx._common_settings, name)
        setattr(yvsx, name, "foo")
        assert getattr(yvsx._common_settings, name) == "foo"


@pytest.fixture(scope="module")
def lines() -> Lines:
    """A Lines plot."""
    return Lines(
        Dataset.from_array(
            array([[0, 0], [1, 1]]),
            variable_names=["x", "y"],
        )
    )


def test_save_plotly(lines, tmp_wd):
    """Check that a plotly-based plot can be saved."""
    lines.execute(file_format="html")
    assert lines.output_files == [str(tmp_wd / "lines.html")]
    assert Path("lines.html").exists()


def test_save_plotly_as_image(lines, tmp_wd):
    """Check that a plotly-based plot can be saved as an image."""
    lines.DEFAULT_PLOT_ENGINE = lines.PlotEngine.PLOTLY
    with mock.patch.object(PlotlyFigure, "write_image") as write_image:
        lines.execute()

    assert write_image.called
    lines.DEFAULT_PLOT_ENGINE = lines.PlotEngine.MATPLOTLIB
    assert write_image.call_args.kwargs == {
        "file": tmp_wd / "lines.png",
        "format": "png",
    }


def test_show_plotly(lines):
    """Check that a plotly-based plot can be displayed."""
    with mock.patch.object(PlotlyFigure, "show") as show:
        lines.execute(save=False, show=True, file_format="html")

    assert show.called


def test_show_matplotlib(lines):
    """Check that a matplotlib-based plot can be displayed."""
    with mock.patch.object(plot, "save_show_figure") as save_show_figure:
        figures = lines.execute(save=False, show=True)

    assert save_show_figure.assert_called_once
    assert save_show_figure.call_args.args == (figures[0], True, "")
