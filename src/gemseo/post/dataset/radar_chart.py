# -*- coding: utf-8 -*-
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
r"""Draw a radar chart from a :class:`.Dataset`. """
from __future__ import division, unicode_literals

from typing import List, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy import linspace, pi, rad2deg

from gemseo.post.dataset.dataset_plot import DatasetPlot


class RadarChart(DatasetPlot):
    """Radar Chart visualization."""

    def _plot(
        self,
        properties,  # type: Mapping
        display_zero=True,  # type: bool
        connect=False,  # type: bool
        radial_ticks=False,  # type: bool
        n_levels=6,  # type: int
        scientific_notation=True,  # type: bool
    ):  # type: (...) -> List[Figure]
        """
        Args:
            display_zero: Whether to display the line where the output is equal to zero.
            connect: Whether to connect the elements of a series with a line.
            radial_ticks: Whether to align the ticks names with the radius.
            n_levels: The number of grid levels.
            scientific_notation: Whether to format the grid levels
                with the scientific notation.
        """
        linestyle = "-o" if connect else "o"

        fig = plt.figure(figsize=self.figsize)
        axe = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")
        axe.grid(True, color="k", linewidth=0.3, linestyle=":")
        axe.tick_params(labelsize=self.font_size)

        all_data, _, sizes = self.dataset.get_all_data(False, False)
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
        if self.color is None:
            colormap = plt.cm.get_cmap(self.colormap)
            self.color = {
                name: colormap(color)
                for name, color in zip(series_names, linspace(0, 1, len(all_data)))
            }

        if self.linestyle is None:
            self.linestyle = {name: linestyle for name in series_names}

        for index, data in enumerate(all_data):
            name = series_names[index]
            data = data.tolist()
            data.append(data[0])
            axe.plot(
                theta,
                data,
                self.linestyle[name],
                color=self.color[name],
                lw=1,
                label=name,
            )

        if display_zero and self.rmin < 0:
            circle = plt.Circle(
                (0, 0),
                abs(self.rmin),
                transform=axe.transData._b,
                fill=False,
                edgecolor="black",
                linewidth=1,
                zorder=10,
            )
            plt.gca().add_artist(circle)

        theta_degree = rad2deg(theta[:-1])
        axe.set_thetagrids(theta_degree, variables_names)
        if radial_ticks:
            labels = []
            for label, angle in zip(axe.get_xticklabels(), theta_degree):
                x, y = label.get_position()
                lab = axe.text(
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
                    angle = angle - 180

                lab.set_rotation(angle)
                labels.append(lab)

            axe.set_xticklabels([])

        axe.set_rlim([self.rmin, self.rmax])
        rticks = linspace(self.rmin, self.rmax, n_levels)
        if scientific_notation:
            rticks_labels = ["{:.2e}".format(value) for value in rticks]
        else:
            rticks_labels = rticks

        axe.set_rticks(rticks)
        axe.set_yticklabels(rticks_labels)
        axe.legend(
            loc="upper left", fontsize=self.font_size, bbox_to_anchor=(1.05, 1.0)
        )
        axe.set_title(self.title, fontsize=self.font_size * 1.2)
        box = axe.get_position()
        axe.set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )
        axe.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=5,
        )
        return [fig]
