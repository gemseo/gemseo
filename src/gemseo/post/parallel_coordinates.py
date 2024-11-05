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
#        :author: Francois Gallard
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A parallel coordinates plot of functions and x."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy import array

from gemseo.post.base_post import BasePost
from gemseo.post.core.colormaps import PARULA
from gemseo.post.parallel_coordinates_settings import ParallelCoordinates_Settings
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from gemseo.typing import NumberArray


class ParallelCoordinates(BasePost[ParallelCoordinates_Settings]):
    """Parallel coordinates plot."""

    Settings: ClassVar[type[ParallelCoordinates_Settings]] = (
        ParallelCoordinates_Settings
    )

    def _plot(self, settings: ParallelCoordinates_Settings) -> None:
        problem = self.optimization_problem

        variable_history, variable_names, _ = self.database.get_history_array(
            function_names=problem.function_names
        )
        names_to_sizes = problem.design_space.variable_sizes
        design_names = [
            repr_variable(name, i, names_to_sizes[name])
            for name in problem.design_space.variable_names
            for i in range(names_to_sizes[name])
        ]
        output_dimension = variable_history.shape[1] - len(design_names)
        design_history = problem.design_space.normalize_vect(
            variable_history[:, output_dimension:]
        )
        function_names = variable_names[:output_dimension]
        objective_index = variable_names.index(self._standardized_obj_name)
        objective_history = variable_history[:, objective_index]
        if self._change_obj:
            objective_history = -objective_history
            variable_history[:, objective_index] = objective_history
            function_names[objective_index] = self._obj_name
            obj_name = self._obj_name
        else:
            obj_name = self._standardized_obj_name

        fig = self.__parallel_coordinates(
            design_history, design_names, objective_history, settings.fig_size
        )
        fig.suptitle(f"Design variables history colored by '{obj_name}' value")
        plt.tight_layout()
        self._add_figure(fig, "para_coord_des_vars")

        fig = self.__parallel_coordinates(
            variable_history[:, :output_dimension],
            function_names,
            objective_history,
            settings.fig_size,
        )
        fig.suptitle(
            f"Objective function and constraints history colored by '{obj_name}' value."
        )
        plt.tight_layout()
        self._add_figure(fig, "para_coord_funcs")

    @classmethod
    def __parallel_coordinates(
        cls,
        y_data: NumberArray,
        x_names: Sequence[str],
        color_criteria: NumberArray,
        fig_size: tuple[float, float],
    ) -> Figure:
        """Plot the parallel coordinates.

        Args:
            y_data: The lines data to plot.
            x_names: The names of the abscissa.
            color_criteria: The values of same length as `y_data`
                to colorize the lines.
            fig_size: The sizes of the figure.
        """
        _, n_cols = y_data.shape
        expected_shape = (len(color_criteria), len(x_names))
        if y_data.shape != expected_shape:
            msg = (
                f"The data shape {y_data.shape} is not equal "
                f"to the expected one {expected_shape}."
            )
            raise ValueError(msg)

        x_values = list(range(n_cols))
        fig = plt.figure(figsize=fig_size)
        ax = plt.gca()
        c_min, c_max = color_criteria.min(), color_criteria.max()
        s_m = matplotlib.cm.ScalarMappable(
            cmap=PARULA, norm=mpl.colors.Normalize(vmin=c_min, vmax=c_max)
        )
        s_m.set_array([])
        color_criteria = (color_criteria - c_min) / (c_max - c_min)
        for i, y_values in enumerate(y_data):
            ax.plot(x_values, y_values, c=array(PARULA(color_criteria[i])))

        for x_value in x_values:
            ax.axvline(x_value, linewidth=1, color="black")

        ax.set_xticks(x_values)
        ax.set_xticklabels(x_names, rotation=90)
        ax.grid()
        fig.colorbar(s_m, ax=ax)
        return fig
