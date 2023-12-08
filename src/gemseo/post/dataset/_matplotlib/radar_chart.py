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
"""Radar chart based on matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from numpy import linspace
from numpy import rad2deg

from gemseo.post.dataset._matplotlib.plot import MatplotlibPlot

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike


class RadarChart(MatplotlibPlot):
    """Radar chart based on matplotlib."""

    def _create_figures(
        self, fig: Figure | None, axes: Axes | None, y_values: ArrayLike, theta: float
    ) -> list[Figure]:
        """
        Args:
            y_values: The values of the series on the y-axis (one series per row).
            theta: The values of the series on the r-axis.
        """  # noqa: D205 D212 D415
        if not fig or not axes:
            fig = plt.figure(figsize=self._common_settings.fig_size)
            axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")

        axes.grid(True, color="k", linewidth=0.3, linestyle=":")
        axes.tick_params(labelsize=self._common_settings.font_size)

        variable_names = self._common_dataset.get_columns()
        series_names = self._common_dataset.index
        for data, name, linestyle, color in zip(
            y_values,
            series_names,
            self._common_settings.linestyle,
            self._common_settings.color,
        ):
            data = data.tolist()
            data.append(data[0])
            axes.plot(
                theta,
                data,
                linestyle,
                color=color,
                lw=1,
                label=name,
            )

        if self._specific_settings.display_zero and self._common_settings.rmin < 0:
            circle = plt.Circle(
                (0, 0),
                abs(self._common_settings.rmin),
                transform=axes.transData._b,
                fill=False,
                edgecolor="black",
                linewidth=1,
                zorder=10,
            )
            plt.gca().add_artist(circle)

        theta_degree = rad2deg(theta[:-1])
        axes.set_thetagrids(theta_degree, variable_names)
        if self._specific_settings.radial_ticks:
            labels = []
            for label, angle in zip(axes.get_xticklabels(), theta_degree):
                x, y = label.get_position()
                lab = axes.text(
                    x,
                    y,
                    label.get_text(),
                    transform=label.get_transform(),
                    ha=label.get_ha(),
                    va=label.get_va(),
                )
                if 90 < angle <= 180:
                    angle = 360 - (180 - angle)

                if 180 < angle < 270:
                    angle -= 180

                lab.set_rotation(angle)
                labels.append(lab)

            axes.set_xticklabels([])

        axes.set_rlim([self._common_settings.rmin, self._common_settings.rmax])
        rticks = linspace(
            self._common_settings.rmin,
            self._common_settings.rmax,
            self._specific_settings.n_levels,
        )
        if self._specific_settings.scientific_notation:
            rticks_labels = [f"{value:.2e}" for value in rticks]
        else:
            rticks_labels = rticks

        axes.set_rticks(rticks)
        axes.set_yticklabels(rticks_labels)
        axes.legend(
            loc="upper left",
            fontsize=self._common_settings.font_size,
            bbox_to_anchor=(1.05, 1.0),
        )
        axes.set_title(
            self._common_settings.title, fontsize=self._common_settings.font_size * 1.2
        )
        box = axes.get_position()
        axes.set_position([
            box.x0,
            box.y0 + box.height * 0.1,
            box.width,
            box.height * 0.9,
        ])
        axes.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=5,
        )
        return [fig]
