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
"""Draw the boxplots of some variables from a dataset.

A boxplot represents the median and the first and third quartiles of numerical data. The
variability outside the inter quartile domain can be represented with lines, called
*whiskers*. The numerical data that are significantly different are called *outliers*
and can be plotted as individual points beyond the whiskers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.boxplot_settings import Boxplot_Settings

if TYPE_CHECKING:
    from collections.abc import Sequence


class Boxplot(BaseDatasetPlot[Boxplot_Settings]):
    """Draw the boxplots of some variables from a dataset."""

    settings_class = Boxplot_Settings

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[Sequence[str], list[float], int, float, list[str], float]:
        """
        Returns:
            The names of the variables,
            the positions of the variables on the x-axis,
            the number of datasets,
            the x-offset,
            the names of the variables,
            the level of opacity.
        """  # noqa: D205, D212, D415
        n_datasets = 1 + len(self.settings.datasets)
        names = self.dataset.get_columns(self.settings.variables)
        self.settings.n_items = n_datasets
        return (
            self.settings.variables,
            [
                (n_datasets - 1) / n_datasets + i * n_datasets
                for i, _ in enumerate(names)
            ],
            n_datasets,
            0,
            names,
            self.settings.opacity_level,
        )
