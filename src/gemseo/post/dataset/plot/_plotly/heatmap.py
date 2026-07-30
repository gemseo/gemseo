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
"""A heat map of a dataset, either rectangular or symmetric, using plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import abs as np_abs
from numpy import isnan
from numpy import log1p
from numpy import sign
from plotly.graph_objects import Heatmap as GOHeatmap

from gemseo.post.dataset.heatmap_settings import Heatmap_Settings
from gemseo.post.dataset.plot._heatmap_utils import compute_centered_bounds
from gemseo.post.dataset.plot._plotly.plot import PlotlyPlot

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

    from gemseo.util.typing import RealArray


class Heatmap(PlotlyPlot[Heatmap_Settings]):
    """A heat map of a dataset, either rectangular or symmetric, using plotly.

    `use_log` is approximated with a signed-log transform of the data
    (`sign(data) * log1p(abs(data))`),
    since plotly has no direct equivalent of matplotlib's `SymLogNorm`.
    When `symmetric` is `True` and `use_log` is also `True`,
    `use_log` takes precedence and the colormap is not centered on `center`.
    """

    def _create_figure(
        self,
        fig: Figure,
        data: RealArray,
        variable_names: tuple[str, ...],
    ) -> Figure:
        """
        Args:
            data: The data to be plotted.
            variable_names: The names of the variables.
        """  # noqa: D205, D212, D415
        settings = self._settings
        if settings.symmetric:
            x = variable_names
            y = variable_names
        else:
            x = tuple(self._common_dataset.index)
            y = variable_names

        z_data = data
        zmid = None
        if settings.use_log:
            z_data = sign(data) * log1p(np_abs(data))
        elif settings.symmetric and settings.center is not None:
            bounds = compute_centered_bounds(data, settings.center)
            zmid = None if bounds is None else settings.center

        text = None
        texttemplate = None
        if settings.annotate:
            text = [
                [
                    "NaN" if isnan(value) else format(value, settings.annotate_fmt)
                    for value in row
                ]
                for row in data
            ]
            texttemplate = "%{text}"  # noqa: RUF027

        fig.add_trace(
            GOHeatmap(
                z=z_data.tolist(),
                x=x,
                y=y,
                colorscale=settings.colormap,
                zmid=zmid,
                opacity=settings.opacity,
                text=text,
                texttemplate=texttemplate,
                colorbar={"title": settings.zlabel},
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
        fig.update_xaxes(showgrid=settings.grid, tickangle=-settings.xtick_rotation)
        fig.update_yaxes(showgrid=settings.grid, autorange="reversed")
        return fig
