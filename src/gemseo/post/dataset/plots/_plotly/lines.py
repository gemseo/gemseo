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
"""Lines based on plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from plotly.graph_objects import Scatter

from gemseo.post.dataset.lines_settings import Lines_Settings
from gemseo.post.dataset.plots._plotly.plot import PlotlyPlot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike
    from plotly.graph_objects import Figure


class Lines(PlotlyPlot[Lines_Settings]):
    """Lines based on plotly."""

    def _create_figure(
        self,
        fig: Figure,
        x_values: ArrayLike,
        y_name_to_value: Mapping[str, ArrayLike],
        default_xlabel: str,
        n_lines: int,
    ) -> Figure:
        """
        Args:
            fig: A Plotly figure.
            x_values: The values on the x-axis.
            y_name_to_value: The variable names bound to the values on the y-axis.
            default_xlabel: The default x-label.
            n_lines: The number of lines.
        """  # noqa: D205 D212 D415
        settings = self._settings
        settings.set_colors(settings.color)
        settings.set_linestyles(settings.linestyle or "-")
        settings.set_markers(settings.marker or "o")
        line_index = -1
        for y_name, y_values in y_name_to_value.items():
            for yi_name, yi_values in zip(
                self._common_dataset.get_columns(y_name), y_values, strict=False
            ):
                line_index += 1
                mode = "lines+markers" if settings.add_markers else "lines"
                fig.add_trace(
                    Scatter(
                        x=list(x_values),
                        y=yi_values,
                        name=yi_name,
                        mode=mode,
                        showlegend=True,
                        line={
                            "dash": self._PLOTLY_LINESTYLES.get(
                                settings.linestyle[line_index], "solid"
                            ),
                            "color": self._stringify_color(settings.color[line_index]),
                            "width": 2,
                        },
                    )
                )
        fig.update_layout(
            title=settings.title,
            xaxis_title=settings.xlabel or default_xlabel,
            yaxis_title=settings.ylabel,
        )
        fig.update_xaxes(showgrid=settings.grid)
        fig.update_yaxes(showgrid=settings.grid)
        if settings.set_xticks_from_data:
            fig.update_layout(xaxis={"tickvals": x_values})

        if settings.use_integer_xticks:
            msg = (
                "The use_integer_xticks option of plotly-based Lines "
                "is not implemented."
            )
            raise NotImplementedError(msg)

        return fig
