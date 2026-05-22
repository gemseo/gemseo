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
r"""Draw a variable versus two others from a [Dataset][gemseo.datasets.dataset.Dataset].

A [ZvsXY][gemseo.post.dataset.zvsxy.ZvsXY]ZvsXY` plot represents the variable $z$
with respect to $x$ and $y$ as a surface plot, based on a set of points
:points $\{x_i,y_i,z_i\}_{1\leq i \leq n}$. This interpolation
relies on the Delaunay triangulation of $\{x_i,y_i\}_{1\leq i \leq n}$
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.datasets.dataset import Dataset  # noqa: TC001
from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.zvsxy_settings import ZvsXY_Settings

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import RealArray


class ZvsXY(BaseDatasetPlot[ZvsXY_Settings]):
    """Plot surface z versus x,y."""

    settings_class = ZvsXY_Settings

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[RealArray, RealArray, RealArray, Iterable[Dataset]]:
        """
        Returns:
            The values of the points on the x-axis,
            the values of the points on the y-axis,
            the values of the points on the z-axis,
            and possibly other datasets.
        """  # noqa: D205, D212, D415
        self.settings.n_items = 1 + len(self.settings.other_datasets)
        x, x_comp = self.settings.x
        y, y_comp = self.settings.y
        z, z_comp = self.settings.z
        if "xlabel" not in self.settings.model_fields_set:
            self.settings.xlabel = self._get_component_name(
                x, x_comp, self.dataset.variable_name_to_n_components
            )
        if "ylabel" not in self.settings.model_fields_set:
            self.settings.ylabel = self._get_component_name(
                y, y_comp, self.dataset.variable_name_to_n_components
            )
        if "zlabel" not in self.settings.model_fields_set:
            self.settings.zlabel = self._get_component_name(
                z, z_comp, self.dataset.variable_name_to_n_components
            )
        if "title" not in self.settings.model_fields_set:
            self.settings.title = self.settings.zlabel
        if "grid" not in self.settings.model_fields_set:
            self.settings.grid = False
        get_view = self.dataset.get_view
        return (
            get_view(variable_names=x, components=x_comp).to_numpy().ravel(),
            get_view(variable_names=y, components=y_comp).to_numpy().ravel(),
            get_view(variable_names=z, components=z_comp).to_numpy().ravel(),
            self.settings.other_datasets,
        )
