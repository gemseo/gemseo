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
"""Draw curves from a [Dataset][gemseo.datasets.dataset.Dataset].

A [Curves][gemseo.post.dataset.curves.Curves] plot represents
samples of a functional variable $y(x)$ discretized over a 1D mesh.
Both evaluations of $y$ and mesh are stored
in a [Dataset][gemseo.datasets.dataset.Dataset],
$y$ as a parameter and the mesh as a misc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.curves_settings import Curves_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Curves(BaseDatasetPlot[Curves_Settings]):
    """Plot curves y_i over the mesh x."""

    settings_class = Curves_Settings

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[RealArray, list[str]]:
        """
        Returns:
            The values of the points of the curves on the y-axis (one curve per row),
            the labels of the curves.
        """  # noqa: D205 D212 D415
        samples = self.settings.samples
        y_values = self.dataset.get_view(
            variable_names=self.settings.variable
        ).to_numpy()
        if samples:
            self.settings.n_items = len(samples)
            y_values = y_values[samples, :]
        else:
            self.settings.n_items = len(y_values)
            samples = range(self.settings.n_items)

        return y_values, [self.dataset.index[sample] for sample in samples]
