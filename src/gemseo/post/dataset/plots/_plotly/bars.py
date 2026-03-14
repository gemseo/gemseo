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
"""A bar plot based on plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import arange
from plotly.graph_objects import Bar

from gemseo.post.dataset.bars_settings import BarPlot_Settings
from gemseo.post.dataset.plots._plotly.plot import PlotlyPlot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray
    from plotly.graph_objects import Figure


class BarPlot(PlotlyPlot[BarPlot_Settings]):
    """A bar plot based on plotly."""

    def _create_figure(
        self,
        fig: Figure,
        data: NDArray,
        feature_names: Iterable[str],
    ) -> Figure:
        """
        Args:
            fig: A Plotly figure.
            data: The data to be plotted.
            feature_names: The names of the features.
        """  # noqa: D205, D212, D415
        settings = self._settings
        settings.set_colors(settings.color)
        n_series, n_features = data.shape
        first_series_positions = arange(n_features)
        width = 0.75 / n_series
        positions = [
            first_series_positions + index * width + width / 2
            for index in range(n_series)
        ]
        for feature_positions, series_name, series_data, series_color in zip(
            positions,
            self._common_dataset.index,
            data,
            settings.color,
            strict=False,
        ):
            text = series_data.tolist() if settings.annotate else None
            fig.add_trace(
                Bar(
                    x=feature_positions,
                    y=series_data.tolist(),
                    width=width,
                    marker={"color": self._stringify_color(series_color)},
                    name=series_name,
                    text=text,
                )
            )
        fig.update_layout(
            font_size=settings.font_size,
            title={
                "text": settings.title,
                "font": {"size": settings.font_size * 1.2},
            },
            xaxis_title=settings.xlabel,
            yaxis_title=settings.ylabel,
        )
        fig.update_xaxes(
            showgrid=settings.grid,
            tickvals=[position + 0.375 for position in first_series_positions],
            ticktext=feature_names,
            tickangle=-settings.xtick_rotation,
        )
        fig.update_yaxes(showgrid=settings.grid)
        fig.update_traces(
            textposition="outside",
            textangle=-settings.annotation_rotation,
        )
        return fig
