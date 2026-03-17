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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Evolution of the variables by means of a color scale."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.color_evolution_settings import ColorEvolution_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class ColorEvolution(BaseDatasetPlot[ColorEvolution_Settings]):
    """Evolution of the variables by means of a color scale.

    Based on the matplotlib function `imshow`.

    Tip:
        Use the colormap setting of the
        [BaseDatasetPlotSettings][gemseo.post.dataset.base_settings.BaseDatasetPlotSettings]
        setting class
        to set a matplotlib colormap, e.g. `"seismic"`.
    """

    settings_class = ColorEvolution_Settings

    def _create_specific_data_from_dataset(self) -> tuple[RealArray, list[str]]:
        """
        Returns:
            The data to be plotted,
            the names of the variables.
        """  # noqa: D205, D212, D415
        return (
            self.dataset.get_view(variable_names=self.settings.variables).to_numpy().T,
            self.settings.variables,
        )
