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
r"""Draw a bar plot from a :class:`.Dataset`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.datasets.dataset import Dataset


class BarPlot(DatasetPlot):
    """Barplot visualization."""

    def __init__(
        self,
        dataset: Dataset,
        n_digits: int = 1,
    ) -> None:
        """
        Args:
            n_digits: The number of digits to print the different bar values.
        """  # noqa: D205, D212, D415
        super().__init__(dataset, n_digits=n_digits)

    def _create_specific_data_from_dataset(self) -> tuple[NDArray[float], list[str]]:
        """
        Returns:
            The data,
            the names of the columns.
        """  # noqa: D205, D212, D415
        data = self.dataset.to_numpy()
        self._n_items = len(data)
        self.colormap = self.colormap
        return data, self.dataset.get_columns()

    @DatasetPlot.colormap.setter
    def colormap(self, value: str) -> None:  # noqa: D102
        self._common_settings.colormap = value
        self._common_settings.color = ""
        self._set_color(self._n_items)
