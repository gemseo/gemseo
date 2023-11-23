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

from typing import TYPE_CHECKING

from numpy import linspace
from numpy import pi

from gemseo.post.dataset.dataset_plot import DatasetPlot

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.datasets.dataset import Dataset


class RadarChart(DatasetPlot):
    """Radar chart visualization."""

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

    def _create_specific_data_from_dataset(self) -> tuple[NDArray[float], list[float]]:
        """
        Returns:
            The values of the series on the y-axis (one series per row),
            the values of the series on the r-axis.
        """  # noqa: D205 D212 D415
        self._n_items = len(self.dataset)
        self.linestyle = "-o" if self._specific_settings.connect else "o"
        y_values = self.dataset.to_numpy()
        self.rmin = y_values.min()
        self.rmax = y_values.max()
        self._set_color(self._n_items)
        dimension = self.dataset.shape[1]
        theta = (2 * pi * linspace(0, 1 - 1.0 / dimension, dimension)).tolist()
        theta.append(theta[0])
        return y_values, theta
