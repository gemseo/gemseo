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

from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.utils.string_tools import repr_variable


class Lines(DatasetPlot):
    """Connect the observations of variables with lines."""

    def __init__(
        self,
        dataset: Dataset,
        variables: Sequence[str] | None = None,
        abscissa_variable: str | None = None,
        add_markers: bool = False,
        set_xticks_from_data: bool = False,
    ) -> None:
        """
        Args:
            variables: The names of the variables to plot.
                If ``None``, use all the variables.
            abscissa_variable: The name of the variable used in abscissa.
                The observations of the ``variables`` are plotted
                in function of the observations of this ``abscissa_variable``.
                If ``None``,
                the observations of the ``variables`` are plotted
                in function of the indices of the observations.
            add_markers: Whether to mark the observations with dots.
            set_xticks_from_data: Whether to use the values of ``abscissa_variable``
                as locations of abscissa ticks.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            variables=variables,
            abscissa_variable=abscissa_variable,
            add_markers=add_markers,
            set_xticks_from_data=set_xticks_from_data,
        )

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        abscissa_variable = self._param.abscissa_variable
        if abscissa_variable is None:
            x_data = range(len(self.dataset))
        else:
            x_data = (
                self.dataset.get_view(variable_names=abscissa_variable)
                .to_numpy()
                .ravel()
                .tolist()
            )

        variables = self._param.variables
        if variables is None:
            variables = self.dataset.variable_names

        y_data = {
            variable_name: self.dataset.get_view(variable_names=variable_name)
            .to_numpy()
            .T
            for variable_name in variables
        }

        n_lines = sum(
            self.dataset.variable_names_to_n_components[name] for name in variables
        )
        self._set_color(n_lines)
        self._set_linestyle(n_lines, "-")
        self._set_marker(n_lines, "o")

        fig, axes = self._get_figure_and_axes(fig, axes)
        line_index = -1
        for variable_name, variable_values in y_data.items():
            variable_size = self.dataset.variable_names_to_n_components[variable_name]
            for variable_component, variable_value in enumerate(variable_values):
                line_index += 1
                axes.plot(
                    x_data,
                    variable_value,
                    linestyle=self.linestyle[line_index],
                    color=self.color[line_index],
                    label=repr_variable(
                        variable_name, variable_component, variable_size
                    ),
                )
                if self._param.add_markers:
                    axes.scatter(
                        x_data,
                        variable_value,
                        color=self.color[line_index],
                        marker=self.marker[line_index],
                    )

        axes.set_xlabel(self.xlabel or abscissa_variable)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)
        axes.legend(loc=self.legend_location)
        if self._param.set_xticks_from_data:
            axes.set_xticks(x_data)
        return [fig]
