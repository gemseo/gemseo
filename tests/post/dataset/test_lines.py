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

import re
from pathlib import Path

import pytest
from matplotlib import pyplot as plt
from numpy import array

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.lines import Lines
from gemseo.utils.testing.helpers import image_comparison


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
    ("kwargs", "properties", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@pytest.mark.parametrize("fig_and_ax", [False, True])
@image_comparison(None)
def test_plot_matplotlib(
    kwargs, properties, baseline_images, dataset, fig_and_ax
) -> None:
    """Test images created by Lines.execute against references for matplotlib."""
    plot = Lines(dataset, **kwargs)
    fig, ax = (None, None) if not fig_and_ax else plt.subplots(figsize=plot.fig_size)
    for k, v in properties.items():
        setattr(plot, k, v)

    plot.execute(save=False, fig=fig, ax=ax)


@pytest.mark.parametrize(
    ("kwargs", "properties", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
def test_plot_plotly(kwargs, properties, baseline_images, dataset) -> None:
    """Test images created by Lines.execute against references for plotly."""
    pytest.importorskip("plotly")
    plot = Lines(dataset, **kwargs)
    for k, v in properties.items():
        setattr(plot, k, v)

    figure = plot.execute(save=False, show=False, file_format="html")[0]
    ref = (
        Path(__file__).parent / "plotly" / "test_lines" / baseline_images[0]
    ).read_text()
    assert figure.to_json() == ref.strip()


def test_pass_existing_figure(dataset):
    """Check that an existing figure can be modified."""
    figure = Lines(dataset, variables="y", abscissa_variable="x").execute(
        save=False, show=False, file_name="existing", file_format="html"
    )[0]
    figure = Lines(
        dataset, variables="z", abscissa_variable="x", add_markers=True
    ).execute(
        save=False, show=False, file_name="final", fig=figure, file_format="html"
    )[0]
    ref = (
        Path(__file__).parent / "plotly" / "test_lines" / "Lines_modified"
    ).read_text()
    assert figure.to_json() == ref.strip()


def test_not_implemented(dataset):
    """Check that the option use_integer_xticks is not implemented with plotly."""
    lines = Lines(
        dataset, variables="y", abscissa_variable="x", use_integer_xticks=True
    )
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "The use_integer_xticks option of plotly-based Lines is not implemented."
        ),
    ):
        lines.execute(save=False, file_format="html")
