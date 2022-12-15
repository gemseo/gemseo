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

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot


class Lines(DatasetPlot):
    """Connect the observations of variables with lines."""

    def __init__(
        self,
        dataset: Dataset,
        variables: Sequence[str] | None = None,
        abscissa_variable: str | None = None,
        add_markers: bool = False,
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
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            variables=variables,
            abscissa_variable=abscissa_variable,
            add_markers=add_markers,
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
            x_data = self.dataset[abscissa_variable].ravel()

        variables = self._param.variables
        if variables is None:
            y_data = self.dataset.get_all_data(False, True)
            variables = y_data.keys()
        else:
            y_data = self.dataset[variables]

        self._set_color(len(variables))
        self._set_linestyle(len(variables), "-")
        self._set_marker(len(variables), "o")

        fig, axes = self._get_figure_and_axes(fig, axes)
        for index, (name, value) in enumerate(y_data.items()):
            axes.plot(
                x_data,
                value,
                linestyle=self.linestyle[index],
                color=self.color[index],
                label=name,
            )
            if self._param.add_markers:
                for sub_value in value.T:
                    axes.scatter(
                        x_data,
                        sub_value,
                        color=self.color[index],
                        marker=self.marker[index],
                    )

        axes.set_xlabel(self.xlabel or abscissa_variable)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)
        axes.legend(loc=self.legend_location)
        return [fig]
