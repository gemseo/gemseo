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
"""Draw a variable versus another from a :class:`.Dataset`.

A :class:`.YvsX` plot represents samples of a couple :math:`(x,y)` as a set of points
whose values are stored in a :class:`.Dataset`. The user can select the style of line or
markers, as well as the color.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.post.dataset.dataset_plot import VariableType

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.datasets.dataset import Dataset


class YvsX(DatasetPlot):
    """Plot curve y versus x."""

    def __init__(self, dataset: Dataset, x: VariableType, y: VariableType) -> None:
        """
        Args:
            x: The name of the variable on the x-axis,
                with its optional component if not ``0``,
                e.g. ``("foo", 3)`` for the fourth component of the variable ``"foo"``.
            y: The name of the variable on the y-axis,
                with its optional component if not ``0``,
                e.g. ``("bar", 3)`` for the fourth component of the variable ``"bar"``.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            x=self._force_variable_to_tuple(x),
            y=self._force_variable_to_tuple(y),
        )

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[NDArray[float], NDArray[float]]:
        """
        Returns:
            The values of the points on the x-axis,
            the values of the points on the y-axis.
        """  # noqa: D205, D212, D415
        x, x_comp = self._specific_settings.x
        y, y_comp = self._specific_settings.y
        self.color = self.color or "blue"
        self.linestyle = self.linestyle or "o"
        variable_names_to_n_components = self.dataset.variable_names_to_n_components
        self.xlabel = self.xlabel or (
            x if variable_names_to_n_components[x] == 1 else f"{x}({x_comp})"
        )
        self.ylabel = self.ylabel or (
            y if variable_names_to_n_components[y] == 1 else f"{y}({y_comp})"
        )
        return (
            self.dataset.get_view(variable_names=x, components=x_comp).to_numpy(),
            self.dataset.get_view(variable_names=y, components=y_comp).to_numpy(),
        )
