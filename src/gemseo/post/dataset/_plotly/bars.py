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
from plotly.graph_objects import Figure

from gemseo.post.dataset._plotly.plot import PlotlyPlot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


class BarPlot(PlotlyPlot):
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
        self._common_settings.set_colors(self._common_settings.color)
        n_series, n_features = data.shape
        first_series_positions = arange(n_features)
        width = 0.75 / n_series
        positions = [
            first_series_positions + index * width + width / 2
            for index in range(n_series)
        ]
        for feature_positions, series_name, series_data, _series_color in zip(
            positions,
            self._common_dataset.index,
            data,
            self._common_settings.color,
        ):
            text = series_data.tolist() if self._specific_settings.annotate else None
            fig.add_trace(
                Bar(
                    x=feature_positions,
                    y=series_data.tolist(),
                    width=width,
                    marker={"color": self._stringify_color(_series_color)},
                    name=series_name,
                    text=text,
                )
            )
        fig.update_layout(
            font_size=self._common_settings.font_size,
            title={
                "text": self._common_settings.title,
                "font": {"size": self._common_settings.font_size * 1.2},
            },
            xaxis_title=self._common_settings.xlabel,
            yaxis_title=self._common_settings.ylabel,
        )
        fig.update_xaxes(
            showgrid=self._common_settings.grid,
            tickvals=[position + 0.375 for position in first_series_positions],
            ticktext=feature_names,
            tickangle=-self._common_settings.xtick_rotation,
        )
        fig.update_yaxes(showgrid=self._common_settings.grid)
        fig.update_traces(
            textposition="outside",
            textangle=-self._specific_settings.annotation_rotation,
        )
        return fig
