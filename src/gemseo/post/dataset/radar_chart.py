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
r"""Draw a radar chart from a [Dataset][gemseo.datasets.dataset.Dataset]."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import linspace
from numpy import pi

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.radar_chart_settings import RadarChart_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class RadarChart(BaseDatasetPlot[RadarChart_Settings]):
    """Radar chart visualization."""

    settings_class = RadarChart_Settings

    def _create_specific_data_from_dataset(self) -> tuple[RealArray, list[float]]:
        """
        Returns:
            The values of the series on the y-axis (one series per row),
            the values of the series on the r-axis.
        """  # noqa: D205 D212 D415
        self.settings.n_items = len(self.dataset)
        if "linestyle" in self.settings.model_fields_set:
            self.settings.set_linestyles(self.settings.linestyle)
        else:
            self.settings.set_linestyles("-o" if self.settings.connect else "o")
        self.settings.set_colors(self.settings.color)
        y_values = self.dataset.to_numpy()
        if "rmin" not in self.settings.model_fields_set:
            self.settings.rmin = y_values.min()
        if "rmax" not in self.settings.model_fields_set:
            self.settings.rmax = y_values.max()
        dimension = self.dataset.shape[1]
        theta = (2 * pi * linspace(0, 1 - 1.0 / dimension, dimension)).tolist()
        theta.append(theta[0])
        return y_values, theta
