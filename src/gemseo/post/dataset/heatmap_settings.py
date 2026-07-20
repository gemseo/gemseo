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

"""The settings for `Heatmap`."""

from __future__ import annotations

from typing import Final

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002
from pydantic import field_validator

from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings

_DEFAULT_MATPLOTLIB_OPTIONS: Final[dict[str, str]] = {
    "interpolation": "nearest",
    "aspect": "auto",
}


class Heatmap_Settings(BaseCartesianDatasetPlotSettings):  # noqa: N801
    """The settings for `Heatmap`."""

    symmetric: bool = Field(
        default=False,
        description="Whether to plot a symmetric variable-versus-variable matrix, "
        "e.g. a correlation matrix or second-order Sobol' indices. "
        "This requires a square dataset, i.e. as many rows as columns. "
        "If `False`, plot the evolution of the variables columns over the entries.",
    )

    variables: tuple[str, ...] = Field(
        default=(),
        description="The variables of interest. If empty, use all the variables.",
    )

    use_log: bool = Field(
        default=False,
        description="Whether to use a symmetric logarithmic scale. "
        "Takes precedence over `center` when `symmetric` is `True`: "
        "the colormap is log-scaled and not centered on `center`.",
    )

    opacity: NonNegativeFloat = Field(
        default=0.6,
        description="The level of opacity (0 = transparent; 1 = opaque).",
        le=1.0,
    )

    center: float | None = Field(
        default=0.0,
        description="The value at which to center a diverging colormap, "
        "e.g. `0.0` for a correlation matrix. "
        "If `None`, do not center the colormap. "
        "Ignored if the data does not take values on both sides of `center`, "
        "or if `use_log` is `True`.",
    )

    annotate: bool = Field(
        default=False, description="Whether to write the value of each cell."
    )

    annotate_fmt: str = Field(
        default=".2g",
        description="The format spec used to render the value of each cell "
        "when `annotate` is `True`.",
    )

    matplotlib_options: dict[str, bool | float | str | None] = Field(
        default_factory=dict,
        description="The options for the matplotlib function `imshow()`. "
        "Default: `interpolation='nearest'` and `aspect='auto'`.",
    )

    @field_validator("matplotlib_options", mode="before")
    @classmethod
    def __validate_matplotlib_options(
        cls, matplotlib_options: dict[str, bool | float | str | None]
    ) -> dict[str, bool | float | str | None]:
        return _DEFAULT_MATPLOTLIB_OPTIONS | matplotlib_options
