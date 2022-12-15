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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Draw a radar chart from a :class:`.Dataset`."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import linspace
from numpy import pi
from numpy import rad2deg

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot


class RadarChart(DatasetPlot):
    """Radar Chart visualization."""

    def __init__(
        self,
        dataset: Dataset,
        display_zero: bool = True,
        connect: bool = False,
        radial_ticks: bool = False,
        n_levels: int = 6,
        scientific_notation: bool = True,
    ) -> None:
        """
        Args:
            display_zero: Whether to display the line where the output is equal to zero.
            connect: Whether to connect the elements of a series with a line.
            radial_ticks: Whether to align the ticks names with the radius.
            n_levels: The number of grid levels.
            scientific_notation: Whether to format the grid levels
                with the scientific notation.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            display_zero=display_zero,
            connect=connect,
            radial_ticks=radial_ticks,
            n_levels=n_levels,
            scientific_notation=scientific_notation,
        )

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        linestyle = "-o" if self._param.connect else "o"

        if not fig or not axes:
            fig = plt.figure(figsize=self.fig_size)
            axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")

        axes.grid(True, color="k", linewidth=0.3, linestyle=":")
        axes.tick_params(labelsize=self.font_size)

        all_data, _, sizes = self.dataset.get_all_data(False)
        variables_names = self.dataset.columns_names
        if self.rmin is None:
            self.rmin = all_data.min()

        if self.rmax is None:
            self.rmax = all_data.max()

        dimension = sum(sizes.values())

        # computes angles
        theta = (2 * pi * linspace(0, 1 - 1.0 / dimension, dimension)).tolist()
        theta.append(theta[0])

        series_names = self.dataset.row_names
        if not self.color:
            colormap = plt.cm.get_cmap(self.colormap)
            self.color = [colormap(color) for color in linspace(0, 1, len(all_data))]

        if not self.linestyle:
            self.linestyle = [linestyle] * len(series_names)

        for data, name, linestyle, color in zip(
            all_data, series_names, self.linestyle, self.color
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

        if self._param.display_zero and self.rmin < 0:
            circle = plt.Circle(
                (0, 0),
                abs(self.rmin),
                transform=axes.transData._b,
                fill=False,
                edgecolor="black",
                linewidth=1,
                zorder=10,
            )
            plt.gca().add_artist(circle)

        theta_degree = rad2deg(theta[:-1])
        axes.set_thetagrids(theta_degree, variables_names)
        if self._param.radial_ticks:
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

        axes.set_rlim([self.rmin, self.rmax])
        rticks = linspace(self.rmin, self.rmax, self._param.n_levels)
        if self._param.scientific_notation:
            rticks_labels = [f"{value:.2e}" for value in rticks]
        else:
            rticks_labels = rticks

        axes.set_rticks(rticks)
        axes.set_yticklabels(rticks_labels)
        axes.legend(
            loc="upper left", fontsize=self.font_size, bbox_to_anchor=(1.05, 1.0)
        )
        axes.set_title(self.title, fontsize=self.font_size * 1.2)
        box = axes.get_position()
        axes.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )
        axes.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=5,
        )
        return [fig]
