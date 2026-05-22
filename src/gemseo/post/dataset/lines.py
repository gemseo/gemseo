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
"""Connect the observations of variables stored in a dataset with lines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.lines_settings import Lines_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Lines(BaseDatasetPlot[Lines_Settings]):
    """Connect the observations of variables with lines."""

    settings_class = Lines_Settings

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[list[float], dict[str, RealArray], str, int]:
        """
        Returns:
            The values on the x-axis,
            the variable names bound to the values on the y-axis,
            the name of the x-label,
            the number of lines.
        """  # noqa: D205 D212 D415
        abscissa_variable = self.settings.abscissa_variable
        if abscissa_variable:
            x_values = (
                self.dataset
                .get_view(variable_names=abscissa_variable)
                .to_numpy()
                .ravel()
                .tolist()
            )
        else:
            x_values = list(range(len(self.dataset)))

        variable_names = list(
            self.settings.variables or self.dataset.columns.levels[1].unique()
        )
        if abscissa_variable:
            if self.settings.plot_abscissa_variable:
                if abscissa_variable not in variable_names:
                    variable_names.append(abscissa_variable)
            elif abscissa_variable in variable_names:
                variable_names.remove(abscissa_variable)

        y_name_to_value = {
            variable_name: self.dataset
            .get_view(variable_names=variable_name)
            .to_numpy()
            .T
            for variable_name in variable_names
        }
        n_lines = sum(
            self.dataset.variable_name_to_n_components[name] for name in variable_names
        )
        self.settings.n_items = n_lines
        return x_values, y_name_to_value, abscissa_variable or "Index", n_lines
