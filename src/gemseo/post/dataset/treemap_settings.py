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

"""The settings for `Treemap`."""

from __future__ import annotations

from matplotlib.colors import is_color_like
from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002
from pydantic import field_validator

from gemseo.post.dataset.base_settings import BaseDatasetPlotSettings

_Color = str | tuple[float, float, float] | tuple[float, float, float, float]
"""A single color: a name or an RGB or RGBA tuple."""


def _normalize_tile_color(
    name: str, color: _Color | tuple[_Color, ...]
) -> tuple[_Color, ...]:
    """Store a tile color as a tuple of colors.

    Args:
        name: The name of the column.
        color: Either a single color or a tuple of colors.

    Returns:
        The tile color as a tuple of colors.

    Raises:
        ValueError: When the value is neither a color nor a tuple of colors.
    """
    if is_color_like(color):
        return (color,)

    if not isinstance(color, str) and all(is_color_like(item) for item in color):
        return tuple(color)

    msg = (
        f"The tile color of {name!r} is neither a color nor a tuple of colors: "
        f"{color!r}."
    )
    raise ValueError(msg)


class Treemap_Settings(BaseDatasetPlotSettings):  # noqa: N801
    """The settings for `Treemap`."""

    colormap: str = Field(
        default="tab10",
        description="""The color map.""",
    )

    tile_colors: dict[str, _Color | tuple[_Color, ...]] = Field(
        default_factory=dict,
        description="The color of each tile, by column name. "
        "Either a single color, "
        "i.e. a name or a tuple of 3 (RGB) or 4 (RGBA) floats, "
        "or a tuple of several colors, "
        "in which case the tile is split into equal bands "
        "along its longer side. "
        "The columns that are missing take the color of the colormap "
        "whose rank is the one of the column in the dataset, "
        "except the residual one, which takes `residual_color`. "
        "Once validated, each value is a tuple of colors: "
        "one item for a single color, several for bands.",
    )

    @field_validator("tile_colors", mode="after")
    @classmethod
    def _normalize_tile_colors(
        cls, tile_colors: dict[str, _Color | tuple[_Color, ...]]
    ) -> dict[str, tuple[_Color, ...]]:
        """Store each tile color as a tuple of colors.

        Args:
            tile_colors: The colors of the tiles,
                either single colors or tuples of colors.

        Returns:
            The colors of the tiles, as tuples of colors.
        """
        return {
            name: _normalize_tile_color(name, color)
            for name, color in tile_colors.items()
        }

    min_area: NonNegativeFloat = Field(
        default=0.0,
        le=1.0,
        description="The threshold on the normalized area of a tile, "
        "i.e. its share of the total area. "
        "The tiles below this threshold are merged into the residual tile.",
    )

    residual_name: str = Field(
        default="",
        description="The name of the column acting as the residual tile, "
        "which is drawn in `residual_color` and always comes last. "
        "If empty, there is no residual tile, "
        "unless tiles are merged because of `min_area`: "
        "a residual tile labelled `residual_label` is then created.",
    )

    residual_label: str = Field(
        default="others",
        description="The label given to the tiles merged because of `min_area`. "
        "It is the label of the residual tile created when `residual_name` is empty; "
        "otherwise, it is appended to the label of the residual tile "
        "as `<label> + <residual_label>`, "
        "unless the label already ends with ` + <residual_label>`.",
    )

    residual_color: str = Field(
        default="lightgrey", description="The color of the residual tile."
    )

    show_values: bool = Field(
        default=True,
        description="Whether to write the value of each tile below its label. "
        "The value is the one of the dataset, "
        "or the sum of the ones of the merged tiles for the residual tile.",
    )

    value_format: str = Field(
        default=".2f",
        description="The format spec used to render the values of the tiles.",
    )

    footnote: str = Field(default="", description="The text under the square.")
