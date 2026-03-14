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
"""The settings for `ScatterMatrix`."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.pyplot import colormaps
from pydantic import Field
from pydantic import PositiveInt  # noqa: TC002
from strenum import StrEnum

from gemseo.post.dataset._trend import Trend
from gemseo.post.dataset._trend import TrendFunctionCreator  # noqa: TC001
from gemseo.post.dataset.base_settings import BaseDatasetPlotSettings

ScatterMatrixOption = bool | int | str | Sequence[str] | None

ColormapName = StrEnum("ColormapName", sorted(colormaps.keys()))


class ScatterMatrix_Settings(BaseDatasetPlotSettings):  # noqa: N801
    """The settings for `ScatterMatrix`."""

    variable_names: tuple[str, ...] = Field(
        default=(),
        description="The variable names to consider. "
        "If empty, consider all the dataset variables.",
    )

    classifier: str = Field(
        default="",
        description="The name of the variable to group data. "
        "If empty, do not group data.",
    )

    kde: bool = Field(
        default=False,
        description="The type of the distribution representation. "
        "If `True`, plot kernel-density estimator on the diagonal. "
        "Otherwise, use histograms.",
    )

    marker: str | tuple[str, ...] = Field(
        default="o",
        description="The marker. "
        "Either a global one or one per item if `n_items` is non-zero. "
        "If empty, use a default one.",
    )

    size: PositiveInt = Field(default=25, description="The size of the points.")

    plot_lower: bool = Field(
        default=True, description="Whether to plot the lower part."
    )

    plot_upper: bool = Field(
        default=True, description="Whether to plot the upper part."
    )

    trend: Trend | TrendFunctionCreator = Field(
        default=Trend.NONE,
        description="The trend function to be added on the scatter plots "
        "or a function creating a trend function from a set of *xy*-points.",
    )

    colormap_name: ColormapName = Field(
        default=ColormapName.cool, description="The name of the matplotlib colormap."
    )

    exclude_classifier: bool = Field(
        default=True,
        description="Whether to exclude the classifier "
        "from the variables to be plotted on the axes.",
    )

    options: dict = Field(
        default_factory=dict,
        description="The options of the underlying pandas scatter matrix.",
    )
