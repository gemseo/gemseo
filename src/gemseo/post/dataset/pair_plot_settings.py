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
"""The settings for `PairPlot`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from matplotlib.pyplot import colormaps
from pydantic import Field
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import model_validator
from strenum import StrEnum

from gemseo.post.dataset.base_settings import BaseDatasetPlotSettings
from gemseo.post.dataset.trend import Trend
from gemseo.post.dataset.trend import TrendFunctionCreator

if TYPE_CHECKING:
    from typing_extensions import Self


ColormapName = StrEnum("ColormapName", sorted(colormaps.keys()))


class PairPlot_Settings(BaseDatasetPlotSettings):  # noqa: N801
    """The settings for `PairPlot`."""

    fig_size: tuple[PositiveFloat, PositiveFloat] = Field(
        default=(6.4, 6.4), description="The figure size."
    )

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

    use_kde: bool = Field(
        default=False,
        description="Whether to plot the marginal probability distributions "
        "using kernel-density estimators (KDE) on the diagonal. "
        "Otherwise, use histograms.",
    )

    use_scatter: bool = Field(
        default=True,
        description="Whether to use scatter plots to represent pairs of variables. "
        "Otherwise, use density surfaces based on kernel density estimators. ",
    )

    marker: str | tuple[str, ...] = Field(
        default=".",
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
        "or a function creating a trend function from a set of *xy*-points. "
        "This option is incompatible with `use_scatter=False`.",
    )

    colormap_name: ColormapName = Field(
        default=ColormapName.cool, description="The name of the matplotlib colormap."
    )

    exclude_classifier: bool = Field(
        default=True,
        description="Whether to exclude the classifier "
        "from the variables to be plotted on the axes.",
    )

    use_ranks: bool = Field(
        default=False,
        description="Whether to plot the component-wise normalized ranks of the data "
        "instead of the raw data on the upper part of the pair plot. "
        "The lower part always shows the raw data. "
        "These ranks define the empirical copula density of the data, "
        "converging towards the true copula. "
        "Thus, "
        "the diagonal represents the marginal distributions, "
        "the lower part represents the joint distribution, "
        "and the upper part represents the copula, "
        "i.e. the dependency structure. "
        "Read [E. Fekhari (2024)](https://theses.hal.science/tel-04617148) "
        "for more details.",
    )

    options: dict[str, Any] = Field(
        default_factory=dict,
        description="The additional keyword arguments passed to "
        "[scatter][matplotlib.axes.Axes.scatter]"
        "or [contour][matplotlib.axes.Axes.contour] "
        "for the off-diagonal cells. "
        "By default, `scatter` uses `alpha=0.5` and `contour` uses `levels=10`.",
    )

    @model_validator(mode="after")
    def __validate(self) -> Self:
        """Validate the settings.

        Raises:
            ValueError: If both `plot_lower` and `plot_upper` are `False`
                or when `use_scatter` is `False` and `trend` is not `Trend.NONE`.
        """
        if not self.plot_lower and not self.plot_upper:
            msg = (
                "At least one of the arguments 'plot_lower' and 'plot_upper' "
                "must be True."
            )
            raise ValueError(msg)

        if self.trend != Trend.NONE and not self.use_scatter:
            msg = (
                "The argument 'trend' must be 'none' "
                "when the argument 'use_scatter' is False."
            )
            raise ValueError(msg)

        return self
