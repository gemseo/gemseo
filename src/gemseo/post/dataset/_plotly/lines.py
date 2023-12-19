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

from plotly.graph_objects import Figure
from plotly.graph_objects import Scatter

from gemseo.post.dataset._plotly.plot import PlotlyPlot

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import ArrayLike


class Lines(PlotlyPlot):
    """Lines based on plotly."""

    def _create_figure(
        self,
        x_values: ArrayLike,
        y_names_to_values: Mapping[str, ArrayLike],
        default_xlabel: str,
    ) -> Figure:
        """
        Args:
            x_values: The values on the x-axis.
            y_names_to_values: The variable names bound to the values on the y-axis.
            default_xlabel: The default x-label.
        """  # noqa: D205 D212 D415
        fig = Figure()
        line_index = -1
        for y_name, y_values in y_names_to_values.items():
            for yi_name, yi_values in zip(
                self._common_dataset.get_columns(y_name), y_values
            ):
                line_index += 1
                mode = (
                    "lines+markers" if self._specific_settings.add_markers else "lines"
                )
                fig.add_trace(
                    Scatter(
                        x=list(x_values),
                        y=yi_values,
                        name=yi_name,
                        mode=mode,
                        showlegend=True,
                        line={
                            "dash": self._PLOTLY_LINESTYLES.get(
                                self._common_settings.linestyle[line_index], "solid"
                            ),
                            "color": self._stringify_color(
                                self._common_settings.color[line_index]
                            ),
                            "width": 2,
                        },
                    )
                )
        fig.update_layout(
            title=self._common_settings.title,
            xaxis_title=self._common_settings.xlabel or default_xlabel,
            yaxis_title=self._common_settings.ylabel,
        )
        if self._specific_settings.set_xticks_from_data:
            fig.update_layout(xaxis={"tickvals": x_values})
        return fig
