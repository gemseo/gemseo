# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import logging
from typing import Sequence

import matplotlib as mpl
from matplotlib import pyplot
from matplotlib.figure import Figure
from numpy import array, ndarray

from gemseo.post.core.colormaps import PARULA
from gemseo.post.opt_post_processor import OptPostProcessor, OptPostProcessorOptionType

LOGGER = logging.getLogger(__name__)


class ParallelCoordinates(OptPostProcessor):
    """Parallel coordinates among design variables, outputs functions and
    constraints."""

    DEFAULT_FIG_SIZE = (10.0, 2.0)

    @classmethod
    def parallel_coordinates(
        cls,
        y_data,  # type: ndarray
        x_names,  # type: Sequence[str]
        color_criteria,  # type: Sequence[float]
    ):  # type: (...) -> Figure
        """Plot the parallel coordinates.

        Args:
            y_data: The lines data to plot.
            x_names: The names of the abscissa.
            color_criteria: The values of same length as `y_data`
                to colorize the lines.
        """
        n_x, n_cols = y_data.shape
        assert n_cols == len(x_names)
        assert n_x == len(color_criteria)
        x_values = list(range(n_cols))
        fig = pyplot.figure(figsize=cls.DEFAULT_FIG_SIZE)
        main_ax = pyplot.gca()
        c_max = color_criteria.max()
        c_min = color_criteria.min()
        color_criteria_n = (color_criteria - c_min) / (c_max - c_min)
        for i in range(n_x):
            color = array(PARULA(color_criteria_n[i])).flatten()
            main_ax.plot(x_values, y_data[i, :], c=color)

        norm = mpl.colors.Normalize(vmin=c_min, vmax=c_max)
        cax, _ = mpl.colorbar.make_axes(main_ax)
        mpl.colorbar.ColorbarBase(cax, norm=norm, cmap=PARULA)
        for i in x_values:
            main_ax.axvline(i, linewidth=1, color="black")

        main_ax.set_xticks(x_values)
        main_ax.set_xticklabels(x_names)
        main_ax.grid()
        return fig

    def _plot(
        self, **options  # type: OptPostProcessorOptionType
    ):  # type: (...) -> None

        all_funcs = self.opt_problem.get_all_functions_names()
        design_variables = self.opt_problem.get_design_variable_names()
        vals, vname, _ = self.database.get_history_array(all_funcs, add_dv=True)
        n_x = len(self.database.get_x_by_iter(0))
        x_names = []
        for d_v in design_variables:
            dv_size = self.opt_problem.design_space.variables_sizes[d_v]
            if dv_size == 1:
                x_names.append(d_v)
            else:
                for i in range(dv_size):
                    x_names.append(d_v + "_" + str(i))
        func_names = vname[: len(vname) - n_x]
        obj_name = self.opt_problem.get_objective_name()
        obj_val = vals[:, vname.index(obj_name)]
        x_vals = vals[:, vals.shape[1] - len(x_names) :]
        x_vals = self.opt_problem.design_space.normalize_vect(x_vals)
        fig = self.parallel_coordinates(x_vals, x_names, obj_val)
        fig.suptitle(
            "Design variables history colored" + " by '" + obj_name + "'  value"
        )

        self._add_figure(fig, "para_coord_des_vars")

        func_vals = vals[:, : vals.shape[1] - len(x_names)]
        fig = self.parallel_coordinates(func_vals, func_names, obj_val)
        fig.suptitle(
            "Objective function and constraints history"
            + " colored by '"
            + obj_name
            + "' value"
        )

        self._add_figure(fig, "para_coord_funcs")
