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
    from gemseo.datasets.dataset import Dataset
    from gemseo.typing import RealArray


class BarPlot(DatasetPlot):
    """Barplot visualization."""

    def __init__(
        self,
        dataset: Dataset,
        n_digits: int = 1,
        annotate: bool = True,
        annotation_rotation: float = 0.0,
    ) -> None:
        """
        Args:
            n_digits: The number of digits to print the different bar values.
            annotate: Whether to add annotations of the height value on each bar.
            annotation_rotation: The angle by which annotations are rotated.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            n_digits=n_digits,
            annotate=annotate,
            annotation_rotation=annotation_rotation,
        )

    def _create_specific_data_from_dataset(self) -> tuple[RealArray, list[str]]:
        """
        Returns:
            The data,
            the names of the columns.
        """  # noqa: D205, D212, D415
        data = self.dataset.to_numpy()
        self._n_items = len(data)
        return data, self.dataset.get_columns()
