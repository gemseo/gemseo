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
from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.post.dataset.yvsx import YvsX
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import array


def test_empty_dataset():
    dataset = Dataset()
    with pytest.raises(ValueError):
        YvsX(dataset, x="x", y="y")


def test_plot_notimplementederror():
    dataset = Dataset()
    dataset.set_from_array(array([[1, 2]]))
    post = DatasetPlot(dataset)
    with pytest.raises(NotImplementedError):
        post._plot()


def test_get_label():
    dataset = Dataset()
    dataset.set_from_array(array([[1, 2]]), variables=["x"], sizes={"x": 2})
    post = DatasetPlot(dataset)
    label, varname = post._get_label(["parameters", "x", 0])
    assert label == "x(0)"
    assert varname == ("parameters", "x", "0")
    error_message = (
        "'variable' must be either a string or a tuple"
        " whose first component is a string and second"
        " one is an integer"
    )
    with pytest.raises(TypeError, match=error_message):
        post._get_label(123)


@pytest.fixture
def plot():
    """A simple dataset plot from a dataset with a single value: x=[1]."""
    dataset = Dataset()
    dataset.set_from_array(array([[1]]), variables=["x"], sizes={"x": 1})
    return DatasetPlot(dataset)


@pytest.mark.parametrize(
    "attribute,default_value",
    [
        ("xlabel", ""),
        ("ylabel", ""),
        ("zlabel", ""),
        ("title", ""),
        ("font_size", 10),
        ("labels", {}),
    ],
)
def test_property_default_values(plot, attribute, default_value):
    """Check the default values of properties."""
    assert getattr(plot, attribute) == default_value


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("xlabel", "dummy_xlabel"),
        ("ylabel", "dummy_ylabel"),
        ("zlabel", "dummy_zlabel"),
        ("title", "dummy_title"),
        ("font_size", 2),
        ("labels", {"x": "dummy_label"}),
    ],
)
def test_setters(plot, attribute, value):
    """Check the attribute setters."""
    setattr(plot, attribute, value)
    assert getattr(plot, attribute) == value


def test_execute_properties(plot):
    """Check that properties are correctly used."""
    plot._plot = lambda fig, axes: []
    plot.execute(save=False, properties={"xlabel": "foo"})
    assert plot.xlabel == "foo"

    with pytest.raises(
        AttributeError, match=r"bar is not an attribute of DatasetPlot\."
    ):
        plot.execute(save=False, properties={"bar": "foo"})


def test_get_figure_and_axes_from_existing_fig_and_axes(plot):
    """Check that get_figure_and_axes using fig and axes returns fig and axes."""
    fig, axes = plt.subplots()
    fig_, axes_ = plot._get_figure_and_axes(fig, axes)
    assert id(fig) == id(fig_)
    assert id(axes) == id(axes_)


def test_get_figure_and_axes_from_scratch(plot):
    """Check that get_figure_and_axes without fig and axes returns new fig and axes."""
    fig, axes = plot._get_figure_and_axes(None, None)
    assert isinstance(fig, Figure)
    assert isinstance(axes, Axes)


def test_get_figure_and_axes_from_axes_only(plot):
    """Check that get_figure_and_axes with axes and without fig raises a ValueError."""
    _, axes = plt.subplots()
    with pytest.raises(
        ValueError, match="The figure associated with the given axes is missing."
    ):
        plot._get_figure_and_axes(None, axes)


def test_get_figure_and_axes_from_figure_only(plot):
    """Check that get_figure_and_axes without axes and with fig raises a ValueError."""
    fig, _ = plt.subplots()

    with pytest.raises(
        ValueError, match="The axes associated with the given figure are missing."
    ):
        plot._get_figure_and_axes(fig, None)
