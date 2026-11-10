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
"""Test the class Treemap."""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from numpy import array
from numpy import nan

from gemseo.dataset import Dataset
from gemseo.dataset.io_dataset import IODataset
from gemseo.post.dataset import Treemap
from gemseo.post.dataset.treemap_settings import Treemap_Settings
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from matplotlib.transforms import Bbox


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """A dataset with a single sample of five variables."""
    return Dataset.from_array(
        array([[0.45, 0.25, 0.15, 0.1, 0.05]]), variable_names=["a", "b", "c", "d", "e"]
    )


@pytest.fixture(scope="module")
def thin_dataset() -> Dataset:
    """A dataset with a single sample, including one tall and thin tile."""
    return Dataset.from_array(
        array([[0.9, 0.06, 0.04]]), variable_names=["big", "thin", "tiny"]
    )


@pytest.mark.parametrize(
    "settings",
    [
        pytest.param(Treemap_Settings(), id="default"),
        pytest.param(
            Treemap_Settings(
                tile_colors={"a": "tab:blue", "b": ("tab:red", "tab:orange")},
                title="The title",
                labels={"a": "Alpha"},
            ),
            id="colors_title_labels",
        ),
        pytest.param(
            Treemap_Settings(min_area=0.12, residual_label="small ones"),
            id="min_area_merged_residual",
        ),
        pytest.param(
            Treemap_Settings(residual_name="e", show_values=False, value_format=".1%"),
            id="residual_name_no_values",
        ),
        pytest.param(
            Treemap_Settings(
                footnote="The footnote", value_format=".1f", colormap="Set2"
            ),
            id="footnote_value_format_colormap",
        ),
    ],
)
def test_plot(settings, dataset, snapshot_matplotlib) -> None:
    """Test images created by Treemap against references."""
    Treemap(dataset, settings).execute(save=False)


def test_plot_rotated_label(thin_dataset, snapshot_matplotlib) -> None:
    """Test the image of a treemap with a tall and thin tile, whose label is rotated."""
    Treemap(thin_dataset, Treemap_Settings(fig_size=(3.0, 6.0))).execute(save=False)


def test_plot_given_fig_and_ax(dataset) -> None:
    """Check that execute() reuses a pre-built fig and ax instead of new ones."""
    settings = Treemap_Settings()
    fig, ax = plt.subplots(figsize=settings.fig_size)
    plot = Treemap(dataset, settings)
    plot.execute(save=False, fig=fig, ax=ax)
    assert plot.figures == [fig]
    # One band and one outline per tile.
    assert len(ax.patches) == 2 * 5


def test_execute_residual_tile_last() -> None:
    """Check that the residual tile is drawn last, with its color."""
    dataset = Dataset.from_array(
        array([[0.5, 0.1, 0.4]]), variable_names=["a", "b", "r"]
    )
    plot = Treemap(dataset, Treemap_Settings(residual_name="r"))
    fig, ax = plt.subplots()
    plot.execute(save=False, fig=fig, ax=ax)
    # One band and one outline per tile, in the order of placement.
    assert to_hex(ax.patches[::2][-1].get_facecolor()) == to_hex("lightgrey")


@pytest.mark.parametrize(
    ("raw_values", "settings", "expected_labels", "expected_values"),
    [
        pytest.param(
            [0.45, 0.25, 0.15, 0.1, 0.05],
            Treemap_Settings(min_area=0.12),
            ("a", "b", "c", "others"),
            (0.45, 0.25, 0.15, 0.15),
            id="merged_tiles",
        ),
        pytest.param(
            [6.0, 2.0, 1.0, 1.0],
            Treemap_Settings(min_area=0.15),
            ("a", "b", "others"),
            (6.0, 2.0, 2.0),
            id="raw_values",
        ),
        pytest.param(
            [0.45, 0.25, 0.15, 0.1, 0.05],
            Treemap_Settings(
                min_area=0.12,
                residual_name="e",
                residual_label="small",
                labels={"e": "Rest"},
            ),
            ("a", "b", "c", "Rest + small"),
            (0.45, 0.25, 0.15, 0.15),
            id="with_merged_tiles",
        ),
        pytest.param(
            [0.45, 0.25, 0.15, 0.1, 0.05],
            Treemap_Settings(residual_name="e"),
            ("a", "b", "c", "d", "e"),
            (0.45, 0.25, 0.15, 0.1, 0.05),
            id="without_merged_tiles",
        ),
        pytest.param(
            [0.45, 0.25, 0.15, 0.1, 0.05],
            Treemap_Settings(
                min_area=0.12,
                residual_name="e",
                residual_label="small",
                labels={"e": "Rest + small"},
            ),
            ("a", "b", "c", "Rest + small"),
            (0.45, 0.25, 0.15, 0.15),
            id="residual_label_suffix_not_repeated",
        ),
        pytest.param(
            [0.5, 0.1, 0.4],
            Treemap_Settings(residual_name="c"),
            ("a", "b", "c"),
            (0.5, 0.1, 0.4),
            id="residual_not_smallest",
        ),
    ],
)
def test_specific_data(raw_values, settings, expected_labels, expected_values) -> None:
    """Check the labels and values of the tiles."""
    dataset = Dataset.from_array(
        array([raw_values]), variable_names=["a", "b", "c", "d", "e"][: len(raw_values)]
    )
    labels, values, _ = Treemap(dataset, settings)._create_specific_data_from_dataset()
    assert labels == expected_labels
    assert values == pytest.approx(expected_values)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([[1.0, 2.0], [3.0, 4.0]], id="several_samples"),
        pytest.param([[1.0, -2.0]], id="negative"),
        pytest.param([[1.0, nan]], id="not_finite"),
        pytest.param([[0.0, 0.0]], id="zero_sum"),
        pytest.param([["a", "b"]], id="not_numeric"),
    ],
)
def test_invalid_dataset_error(values, snapshot) -> None:
    """Check that an error is raised when the dataset is invalid."""
    with assert_exception(ValueError, snapshot):
        Treemap(Dataset.from_array(array(values)))


def test_unknown_residual_error(dataset, snapshot) -> None:
    """Check that an error is raised when the residual tile is not a column."""
    with assert_exception(ValueError, snapshot):
        Treemap(dataset, Treemap_Settings(residual_name="z"))


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        pytest.param(
            Treemap_Settings(min_area=0.12), ("lightgrey",), id="merged_tiles"
        ),
        pytest.param(
            Treemap_Settings(residual_name="e", tile_colors={"e": "red"}),
            ("red",),
            id="color_of_the_column",
        ),
        pytest.param(
            Treemap_Settings(residual_name="e", tile_colors={"e": (0.1, 0.2, 0.3)}),
            ((0.1, 0.2, 0.3),),
            id="rgb_color_of_the_column",
        ),
        pytest.param(
            Treemap_Settings(
                residual_name="e", tile_colors={"e": (0.1, 0.2, 0.3, 0.5)}
            ),
            ((0.1, 0.2, 0.3, 0.5),),
            id="rgba_color_of_the_column",
        ),
        pytest.param(
            Treemap_Settings(residual_name="e", tile_colors={"e": ("red", "blue")}),
            ("red", "blue"),
            id="bands_of_the_column",
        ),
        pytest.param(
            Treemap_Settings(residual_name="e", residual_color="blue"),
            ("blue",),
            id="residual_color",
        ),
    ],
)
def test_specific_data_residual_color(dataset, settings, expected) -> None:
    """Check the color of the residual tile."""
    colors = Treemap(dataset, settings)._create_specific_data_from_dataset()[2]
    assert colors[-1] == expected


@pytest.mark.parametrize(
    "color",
    [
        pytest.param("kb", id="string_of_color_letters"),
        pytest.param("gren", id="misspelled_name"),
        pytest.param((255, 0, 0), id="rgb_out_of_range"),
    ],
)
def test_invalid_tile_color_error(color, snapshot) -> None:
    """Check that an invalid tile color is rejected when building the settings."""
    with assert_exception(ValueError, snapshot):
        Treemap_Settings(tile_colors={"a": color})


def test_duplicated_column_names_error(snapshot) -> None:
    """Check that an error is raised when two columns have the same name."""
    dataset = IODataset()
    dataset.add_variable("y", array([[1.0]]), group_name=dataset.INPUT_GROUP)
    dataset.add_variable("y", array([[3.0]]), group_name=dataset.OUTPUT_GROUP)
    with assert_exception(ValueError, snapshot):
        Treemap(dataset)


def _get_text_boxes(figure: Figure) -> list[Bbox]:
    """Get the boxes of the backgrounds of the written texts.

    Args:
        figure: The figure, already drawn.

    Returns:
        The boxes of the backgrounds of the texts, in pixels.
    """
    renderer = figure.canvas.get_renderer()
    ax = figure.axes[0]
    return [
        text.get_bbox_patch().get_window_extent(renderer)
        for text in ax.texts
        if text.get_bbox_patch() is not None and text.get_text()
    ]


def _assert_texts_inside_tiles(figure: Figure) -> None:
    """Assert that each written text is inside a tile.

    Args:
        figure: The figure, already drawn.
    """
    renderer = figure.canvas.get_renderer()
    tiles = [patch.get_window_extent(renderer) for patch in figure.axes[0].patches]
    boxes = _get_text_boxes(figure)
    # The texts that do not fit in their tile are not written.
    assert boxes
    for box in boxes:
        assert any(
            tile.x0 <= box.x0
            and box.x1 <= tile.x1
            and tile.y0 <= box.y0
            and box.y1 <= tile.y1
            for tile in tiles
        )


def test_execute_text_inside_tiles(dataset) -> None:
    """Check that the texts fit in their tiles when the figure is not square."""
    # Labels that would fit if measured against the width of the axes
    # before the equal aspect shrinks them.
    labels = {name: f"{name}_a_rather_long_label" for name in dataset.variable_names}
    plot = Treemap(dataset, Treemap_Settings(fig_size=(10.0, 3.0), labels=labels))
    plot.execute(save=False)
    figure = plot.figures[0]
    figure.canvas.draw()
    _assert_texts_inside_tiles(figure)


@pytest.mark.parametrize("file_format", ["svg", "pdf", "png"])
def test_save_with_any_backend(dataset, file_format) -> None:
    """Check that a treemap in a plain figure can be saved with any backend."""
    figure = Figure()
    plot = Treemap(dataset)
    plot.execute(save=False, fig=figure, ax=figure.subplots())
    buffer = BytesIO()
    plot.figures[0].savefig(buffer, format=file_format)
    assert buffer.getvalue()


def test_text_fits_after_resizing(dataset) -> None:
    """Check that the texts are fitted again when the figure is resized."""
    plot = Treemap(dataset, Treemap_Settings(fig_size=(10.0, 10.0)))
    plot.execute(save=False)
    figure = plot.figures[0]
    figure.canvas.draw()
    figure.set_size_inches(3.0, 3.0)
    figure.canvas.draw()
    _assert_texts_inside_tiles(figure)


def test_execute_zero_value() -> None:
    """Check that a zero value gets no tile and no text."""
    dataset = Dataset.from_array(
        array([[0.6, 0.0, 0.4]]), variable_names=["a", "z", "b"]
    )
    plot = Treemap(dataset)
    plot.execute(save=False)
    figure = plot.figures[0]
    figure.canvas.draw()
    axes = figure.axes[0]
    outlines = [patch for patch in axes.patches if not patch.get_fill()]
    assert len(outlines) == 2
    assert sum(patch.get_width() * patch.get_height() for patch in outlines) == (
        pytest.approx(1.0)
    )
    labels = [text.get_text().split("\n")[0] for text in axes.texts]
    assert sorted(labels) == ["a", "b"]
