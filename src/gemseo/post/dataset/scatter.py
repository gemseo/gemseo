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
r"""Draw a scatter plot from a :class:`.Dataset`.

A :class:`.Scatter` plot represents a set of points
:math:`\{x_i,y_i\}_{1\leq i \leq n}` as markers on a classical plot
where the color of points can be heterogeneous.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.dataset_plot import VariableType

from gemseo.post.dataset._trend import Trend as _Trend
from gemseo.post.dataset._trend import TrendFunctionCreator


class Scatter(DatasetPlot):
    """Plot curve y versus x."""

    Trend = _Trend
    """The type of trend."""

    def __init__(
        self,
        dataset: Dataset,
        x: VariableType,
        y: VariableType,
        trend: Trend | TrendFunctionCreator = Trend.NONE,
    ) -> None:
        """
        Args:
            x: The name of the variable on the x-axis,
                with its optional component if not ``0``,
                e.g. ``("foo", 3)`` for the fourth component of the variable ``"foo"``.
            y: The name of the variable on the y-axis,
                with its optional component if not ``0``,
                e.g. ``("bar", 3)`` for the fourth component of the variable ``"bar"``.
            trend: The trend function to be added on the scatter plots
                or a function creating a trend function from a set of *xy*-points.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            x=self._force_variable_to_tuple(x),
            y=self._force_variable_to_tuple(y),
            trend=trend,
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
        x_values = self.dataset.get_view(variable_names=x, components=x_comp).to_numpy()
        y_values = self.dataset.get_view(variable_names=y, components=y_comp).to_numpy()
        if self.dataset.variable_names_to_n_components[x] == 1:
            self.xlabel = self.xlabel or x
        else:
            self.xlabel = self.xlabel or f"{x}({x_comp})"

        if self.dataset.variable_names_to_n_components[y] == 1:
            self.ylabel = self.ylabel or y
        else:
            self.ylabel = self.ylabel or f"{y}({y_comp})"

        return x_values, y_values
