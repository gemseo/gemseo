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
r"""Draw a bar plot from a [Dataset][gemseo.datasets.dataset.Dataset]."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.bars_settings import BarPlot_Settings
from gemseo.post.dataset.base import BaseDatasetPlot

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class BarPlot(BaseDatasetPlot[BarPlot_Settings]):
    """Barplot visualization."""

    settings_class = BarPlot_Settings

    def _create_specific_data_from_dataset(self) -> tuple[RealArray, list[str]]:
        """
        Returns:
            The data,
            the names of the columns.
        """  # noqa: D205, D212, D415
        data = self.dataset.to_numpy()
        self.settings.n_items = len(data)
        return data, self.dataset.get_columns()
