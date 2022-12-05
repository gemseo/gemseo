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

from typing import Sequence

import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import array
from numpy import ndarray

from gemseo.post.core.colormaps import PARULA
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.post.opt_post_processor import OptPostProcessorOptionType


class ParallelCoordinates(OptPostProcessor):
    """Parallel coordinates among design variables, outputs functions and constraints."""

    DEFAULT_FIG_SIZE = (10.0, 5.0)

    @classmethod
    def parallel_coordinates(
        cls,
        y_data: ndarray,
        x_names: Sequence[str],
        color_criteria: Sequence[float],
    ) -> Figure:
        """Plot the parallel coordinates.

        Args:
            y_data: The lines data to plot.
            x_names: The names of the abscissa.
            color_criteria: The values of same length as `y_data`
                to colorize the lines.
        """
        n_x, n_cols = y_data.shape
        expected_shape = (len(color_criteria), len(x_names))
        if y_data.shape != expected_shape:
            raise ValueError(
                f"The data shape {y_data.shape} is not equal "
                f"to the expected one {expected_shape}."
            )

        x_values = list(range(n_cols))
        fig = plt.figure(figsize=cls.DEFAULT_FIG_SIZE)
        axes = plt.gca()
        c_min, c_max = color_criteria.min(), color_criteria.max()
        s_m = matplotlib.cm.ScalarMappable(
            cmap=PARULA, norm=mpl.colors.Normalize(vmin=c_min, vmax=c_max)
        )
        s_m.set_array([])
        color_criteria = (color_criteria - c_min) / (c_max - c_min)
        for i, y_values in enumerate(y_data):
            axes.plot(x_values, y_values, c=array(PARULA(color_criteria[i])))

        for x_value in x_values:
            axes.axvline(x_value, linewidth=1, color="black")

        axes.set_xticks(x_values)
        axes.set_xticklabels(x_names, rotation=90)
        axes.grid()
        fig.colorbar(s_m, ax=axes)
        return fig

    def _plot(self, **options: OptPostProcessorOptionType) -> None:
        problem = self.opt_problem
        variable_history, variable_names, _ = self.database.get_history_array(
            problem.get_all_functions_names()
        )
        n_x = len(self.database.get_x_by_iter(0))
        design_names = variable_names[len(variable_names) - n_x :]
        design_history = variable_history[:, len(variable_names) - n_x :]
        design_history = problem.design_space.normalize_vect(design_history)
        function_names = variable_names[: len(variable_names) - n_x]
        function_history = variable_history[:, : len(variable_names) - n_x]

        objective_index = variable_names.index(self._standardized_obj_name)
        objective_history = variable_history[:, objective_index]
        if self._change_obj:
            objective_history = -objective_history
            variable_history[:, objective_index] = objective_history
            function_names[objective_index] = self._obj_name
            obj_name = self._obj_name
        else:
            obj_name = self._standardized_obj_name

        fig = self.parallel_coordinates(design_history, design_names, objective_history)
        fig.suptitle(f"Design variables history colored by '{obj_name}' value")
        plt.tight_layout()
        self._add_figure(fig, "para_coord_des_vars")

        fig = self.parallel_coordinates(
            function_history, function_names, objective_history
        )
        fig.suptitle(
            f"Objective function and constraints history "
            f"colored by '{obj_name}' value."
        )
        plt.tight_layout()
        self._add_figure(fig, "para_coord_funcs")
