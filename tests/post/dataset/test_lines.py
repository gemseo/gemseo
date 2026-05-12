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
from gemseo.post.dataset.lines import Lines
from gemseo.post.dataset.lines_settings import Lines_Settings
from gemseo.utils.testing.helpers import assert_exception


@pytest.fixture(scope="module")
def dataset():
    """Dataset: A dataset containing 3 samples of variables x, y and z (dim(z)=2)."""
    sample1 = [0.0, 1.0, 1.0, 0.0]
    sample2 = [0.75, 0.0, 0.5, 1.0]
    sample3 = [1.0, 1.0, 0.75, 0.0]
    data_array = array([sample1, sample2, sample3])
    variable_names_to_n_components = {"x": 1, "y": 1, "z": 2}
    return Dataset.from_array(
        data_array,
        variable_names=["x", "y", "z"],
        variable_names_to_n_components=variable_names_to_n_components,
    )


TEST_PARAMETERS = {
    "default": ({}, {}, ["Lines"]),
    "xticks": ({"set_xticks_from_data": True}, {}, ["Lines_xticks"]),
    "markers": ({"add_markers": True}, {}, ["Lines_markers"]),
    "variables": ({"variables": ["y"]}, {}, ["Lines_variables"]),
    "abscissa": (
        {"abscissa_variable": "x"},
        {},
        ["Lines_abscissa"],
    ),
    "plot_abscissa_variable_1": (
        {"abscissa_variable": "x", "plot_abscissa_variable": True},
        {},
        ["Lines_plot_abscissa_variable_1"],
    ),
    "plot_abscissa_variable_2": (
        {"abscissa_variable": "x", "plot_abscissa_variable": True, "variables": ["y"]},
        {},
        ["Lines_plot_abscissa_variable_2"],
    ),
    "with_properties": (
        {"add_markers": True},
        {
            "xlabel": "The xlabel",
            "ylabel": "The ylabel",
            "title": "The title",
            "legend_location": "upper right",
            "color": ["red", "black", "blue", "blue"],
            "linestyle": ["--", "-.", "-", "-"],
            "marker": ["o", "v", "<", "<"],
            "grid": False,
        },
        ["Lines_properties"],
    ),
    "with_other_properties": (
        {"add_markers": True},
        {"color": "red", "linestyle": "--", "marker": "v"},
        ["Lines_other_properties"],
    ),
    "colormap": (
        {"add_markers": True},
        {"colormap": "viridis"},
        ["Lines_colormap"],
    ),
}


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [v[:2] for v in TEST_PARAMETERS.values()],
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
def test_plot_matplotlib(
    kwargs, properties, dataset, fig_and_ax, snapshot_matplotlib
) -> None:
    """Test images created by Lines.execute against references for matplotlib."""
    settings = Lines_Settings(**kwargs, **properties)
    plot = Lines(dataset, settings)
    fig, ax = (
        (None, None) if not fig_and_ax else plt.subplots(figsize=settings.fig_size)
    )
    plot.execute(save=False, fig=fig, ax=ax)


@pytest.mark.parametrize(
    ("kwargs", "properties"),
    [v[:2] for v in TEST_PARAMETERS.values()],
    ids=TEST_PARAMETERS.keys(),
)
def test_plot_plotly(kwargs, properties, snapshot, dataset) -> None:
    """Test images created by Lines.execute against references for plotly."""
    pytest.importorskip("plotly")
    settings = Lines_Settings(**kwargs, **properties)
    plot = Lines(dataset, settings)
    figure = plot.execute(save=False, file_format="html")[0]
    assert figure.to_json() == snapshot


def test_pass_existing_figure(dataset, snapshot):
    """Check that an existing figure can be modified."""
    settings = Lines_Settings(variables=["y"], abscissa_variable="x")
    figure = Lines(dataset, settings).execute(
        save=False, file_name="existing", file_format="html"
    )[0]
    settings = Lines_Settings(variables=["z"], abscissa_variable="x", add_markers=True)
    figure = Lines(dataset, settings).execute(
        save=False, file_name="final", fig=figure, file_format="html"
    )[0]
    assert figure.to_json() == snapshot


def test_not_implemented(dataset, snapshot):
    """Check that the option use_integer_xticks is not implemented with plotly."""
    settings = Lines_Settings(
        variables=["y"], abscissa_variable="x", use_integer_xticks=True
    )
    lines = Lines(dataset, settings)
    with assert_exception(NotImplementedError, snapshot):
        lines.execute(save=False, file_format="html")
