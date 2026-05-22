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
r"""Draw a scatter plot from a [Dataset][gemseo.datasets.dataset.Dataset].

A [Scatter][gemseo.post.dataset.scatter.Scatter] plot represents a set of points
$\{x_i,y_i\}_{1\leq i \leq n}$ as markers on a classical plot
where the color of points can be heterogeneous.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.scatter_settings import Scatter_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Scatter(BaseDatasetPlot[Scatter_Settings]):
    """Plot curve y versus x."""

    settings_class = Scatter_Settings

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[RealArray, RealArray]:
        """
        Returns:
            The values of the points on the x-axis,
            the values of the points on the y-axis.
        """  # noqa: D205, D212, D415
        x, x_comp = self.settings.x
        y, y_comp = self.settings.y
        self.settings.set_colors(self.settings.color or "blue")
        x_values = self.dataset.get_view(variable_names=x, components=x_comp).to_numpy()
        y_values = self.dataset.get_view(variable_names=y, components=y_comp).to_numpy()
        if "xlabel" not in self.settings.model_fields_set:
            if self.dataset.variable_name_to_n_components[x] == 1:
                self.settings.xlabel = x
            else:
                self.settings.xlabel = f"{x}({x_comp})"

        if "ylabel" not in self.settings.model_fields_set:
            if self.dataset.variable_name_to_n_components[y] == 1:
                self.settings.ylabel = y
            else:
                self.settings.ylabel = f"{y}({y_comp})"

        return x_values, y_values
