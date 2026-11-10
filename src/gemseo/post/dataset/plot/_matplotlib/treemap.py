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
"""A squarified treemap of the variables of a dataset, using matplotlib."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from matplotlib.patches import Rectangle
from matplotlib.text import Text

from gemseo.post.dataset.plot._matplotlib.plot import MatplotlibPlot
from gemseo.post.dataset.plot._treemap_layout import compute_squarified_layout
from gemseo.post.dataset.treemap_settings import Treemap_Settings

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.backend_bases import RendererBase
    from matplotlib.figure import Figure

    from gemseo.post.dataset.treemap_settings import _Color

_minimum_font_size: Final[int] = 5
"""The smallest font size of the text written on a tile."""

_filling_ratio: Final[float] = 0.9
"""The maximum share of the width and height of a tile that its text can take."""

_elongation_ratio: Final[float] = 2.0
"""The height-to-width ratio above which the text of a tile can be rotated."""

_text_padding: Final[float] = 0.3
"""The padding around the text of a tile, as a fraction of its font size."""

_text_background: Final[Mapping[str, str | float]] = MappingProxyType({
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.7,
})
"""The style of the background of the text of a tile, except its box style."""


class Treemap(MatplotlibPlot[Treemap_Settings]):
    """A squarified treemap of the variables of a dataset, using matplotlib."""

    def _create_figures(
        self,
        fig: Figure | None,
        ax: Axes | None,
        display_labels: Sequence[str],
        display_values: Sequence[float],
        colors: Sequence[Sequence[_Color]],
    ) -> list[Figure]:
        """
        Args:
            display_labels: The labels of the tiles.
            display_values: The values to be written on the tiles.
            colors: The colors of the tiles, with one or several colors per tile.
        """  # noqa: D205, D212, D415
        settings = self._settings
        fig, ax = self._get_figure_and_axes(fig, ax)
        ax.set_xlim(0.0, 1.0)
        # The origin is at the top left corner, where the largest tile is.
        ax.set_ylim(1.0, 0.0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(settings.title)
        if settings.footnote:
            ax.text(
                0.5,
                -0.01,
                settings.footnote,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=max(settings.font_size - 2, _minimum_font_size),
                wrap=True,
            )

        # The tiles are already ordered, with the residual one last.
        tiles = [
            tile
            for tile in zip(
                compute_squarified_layout(display_values, 1.0, 1.0),
                display_labels,
                display_values,
                colors,
                strict=True,
            )
            if tile[0].width > 0 and tile[0].height > 0
        ]
        for rectangle, label, value, tile_colors in tiles:
            self.__draw_tile(ax, rectangle, tile_colors)
            self.__add_text(ax, rectangle, label, value)

        return [fig]

    @staticmethod
    def __draw_tile(
        ax: Axes,
        rectangle: tuple[float, float, float, float],
        colors: Sequence[_Color],
    ) -> None:
        """Draw a tile, split into bands along its longer side if it has many colors.

        The bands have no edge,
        so that the outline of the tile separates it from the other tiles only.

        Args:
            ax: The axes.
            rectangle: The abscissa, ordinate, width and height of the tile.
            colors: The colors of the bands.
        """
        x, y, width, height = rectangle
        n_bands = len(colors)
        for index, color in enumerate(colors):
            if width >= height:
                band_width = width / n_bands
                origin = (x + index * band_width, y)
                size = (band_width, height)
            else:
                band_height = height / n_bands
                origin = (x, y + index * band_height)
                size = (width, band_height)

            ax.add_patch(Rectangle(origin, *size, facecolor=color, edgecolor="none"))

        ax.add_patch(
            Rectangle(
                (x, y), width, height, fill=False, edgecolor="white", linewidth=1.5
            )
        )

    def __add_text(
        self,
        ax: Axes,
        rectangle: tuple[float, float, float, float],
        label: str,
        value: float,
    ) -> None:
        """Add the label and value of a tile, to be fitted to it when drawing.

        Args:
            ax: The axes.
            rectangle: The abscissa, ordinate, width and height of the tile.
            label: The label of the tile.
            value: The value to be written on the tile.
        """
        settings = self._settings
        if settings.show_values:
            value_as_text = format(value, settings.value_format)
            texts = (f"{label}\n{value_as_text}", f"{label}: {value_as_text}")
        else:
            texts = (label,)

        x, y, width, height = rectangle
        ax.add_artist(
            _FittedText(
                rectangle,
                texts,
                settings.font_size,
                x + width / 2,
                y + height / 2,
                "",
                transform=ax.transData,
                ha="center",
                va="center",
                color="black",
                bbox={"boxstyle": f"round,pad={_text_padding}", **_text_background},
            )
        )


class _FittedText(Text):
    """A text fitted to a tile of the treemap each time it is drawn.

    The text is written with the largest font that fits in the tile.
    It is not drawn if it does not fit, whatever the font size.
    As the fit is computed when drawing,
    it stays valid when the figure is resized or saved with another size or backend.
    """

    def __init__(
        self,
        rectangle: tuple[float, float, float, float],
        texts: tuple[str, ...],
        font_size: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            rectangle: The abscissa, ordinate, width and height of the tile.
            texts: The variants of the text, by decreasing preference.
            font_size: The largest font size.
            *args: The positional arguments of [Text][matplotlib.text.Text].
            **kwargs: The keyword arguments of [Text][matplotlib.text.Text].
        """  # noqa: D205, D212, D415
        super().__init__(*args, **kwargs)
        self.__rectangle = rectangle
        self.__texts = texts
        self.__font_size = font_size
        self.__cache: dict[tuple[float, float, float], tuple[int, str, int] | None] = {}

    def draw(self, renderer: RendererBase) -> None:  # noqa: D102
        points_to_pixels = renderer.points_to_pixels(1.0)
        x, y, width, height = self.__rectangle
        (left, top), (right, bottom) = self.get_transform().transform([
            (x, y),
            (x + width, y + height),
        ])
        available_width = abs(right - left) * _filling_ratio
        available_height = abs(bottom - top) * _filling_ratio
        key = (available_width, available_height, points_to_pixels)
        if key not in self.__cache:
            self.__cache[key] = self.__fit(
                renderer, points_to_pixels, available_width, available_height
            )

        fit = self.__cache[key]
        if fit is None:
            # An empty text is not drawn, nor is its background.
            self.set_text("")
            return

        font_size, text, rotation = fit
        self.set_fontsize(font_size)
        self.set_text(text)
        self.set_rotation(rotation)
        super().draw(renderer)

    def __fit(
        self,
        renderer: RendererBase,
        points_to_pixels: float,
        available_width: float,
        available_height: float,
    ) -> tuple[int, str, int] | None:
        """Find the largest font size and the variant of the text that fit the tile.

        The extent of a text is proportional to its font size,
        apart from the padding of its background,
        so each variant is measured once.

        Args:
            renderer: The renderer used to measure the text.
            points_to_pixels: The number of pixels per point.
            available_width: The width available for the text, in pixels.
            available_height: The height available for the text, in pixels.

        Returns:
            The font size, the text and its rotation if the text fits;
            otherwise, `None`.
        """
        variants = [(text, 0) for text in self.__texts]
        if available_height > _elongation_ratio * available_width:
            variants += [(text, 90) for text in self.__texts]

        self.set_fontsize(self.__font_size)
        padding_per_size = 2 * _text_padding * points_to_pixels
        best: tuple[int, str, int] | None = None
        for text, rotation in variants:
            self.set_text(text)
            self.set_rotation(rotation)
            extent = super().get_window_extent(renderer)
            font_size = min(
                self.__font_size,
                int(
                    min(
                        available_width
                        / (extent.width / self.__font_size + padding_per_size),
                        available_height
                        / (extent.height / self.__font_size + padding_per_size),
                    )
                ),
            )
            # Check the linear estimate, as the rendering of text is not exactly so.
            while font_size >= _minimum_font_size:
                self.set_fontsize(font_size)
                extent = super().get_window_extent(renderer)
                padding = padding_per_size * font_size
                if (
                    extent.width + padding <= available_width
                    and extent.height + padding <= available_height
                ):
                    break

                font_size -= 1
            else:
                continue

            if best is None or font_size > best[0]:
                best = (font_size, text, rotation)

        return best
