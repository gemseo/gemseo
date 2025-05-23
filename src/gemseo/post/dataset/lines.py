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
"""Connect the observations of variables stored in a :class:`.Dataset` with lines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.datasets.dataset import Dataset
    from gemseo.typing import RealArray

from gemseo.post.dataset.dataset_plot import DatasetPlot


class Lines(DatasetPlot):
    """Connect the observations of variables with lines."""

    def __init__(
        self,
        dataset: Dataset,
        variables: Sequence[str] = (),
        abscissa_variable: str = "",
        add_markers: bool = False,
        set_xticks_from_data: bool = False,
        use_integer_xticks: bool = False,
        plot_abscissa_variable: bool = False,
    ) -> None:
        """
        Args:
            variables: The names of the variables to plot.
                If empty, use all the variables.
            abscissa_variable: The name of the variable used in abscissa.
                The observations of the ``variables`` are plotted
                in function of the observations of this ``abscissa_variable``.
                If empty,
                the observations of the ``variables`` are plotted
                in function of the indices of the observations.
            add_markers: Whether to mark the observations with dots.
            set_xticks_from_data: Whether to use the values of ``abscissa_variable``
                as locations of abscissa ticks.
            use_integer_xticks: Whether to use integer xticks.
            plot_abscissa_variable: Whether to plot the abscissa variable.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            variables=variables,
            abscissa_variable=abscissa_variable,
            add_markers=add_markers,
            set_xticks_from_data=set_xticks_from_data,
            use_integer_xticks=use_integer_xticks,
            plot_abscissa_variable=plot_abscissa_variable,
        )

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
        abscissa_variable = self._specific_settings.abscissa_variable
        if abscissa_variable:
            x_values = (
                self.dataset.get_view(variable_names=abscissa_variable)
                .to_numpy()
                .ravel()
                .tolist()
            )
        else:
            x_values = list(range(len(self.dataset)))

        variable_names = list(
            self._specific_settings.variables or self.dataset.variable_names
        )
        if abscissa_variable:
            if self._specific_settings.plot_abscissa_variable:
                if abscissa_variable not in variable_names:
                    variable_names.append(abscissa_variable)
            elif abscissa_variable in variable_names:
                variable_names.remove(abscissa_variable)

        y_names_to_values = {
            variable_name: self.dataset.get_view(variable_names=variable_name)
            .to_numpy()
            .T
            for variable_name in variable_names
        }
        n_lines = sum(
            self.dataset.variable_names_to_n_components[name] for name in variable_names
        )
        self._n_items = n_lines
        return x_values, y_names_to_values, abscissa_variable or "Index", n_lines
