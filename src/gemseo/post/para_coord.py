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
"""
A parallel coordinates plot of functions and x
**********************************************
"""

from __future__ import absolute_import, division, unicode_literals

from os.path import splitext

import matplotlib as mpl
from future import standard_library
from matplotlib import pyplot
from numpy import array

from gemseo.post.core.colormaps import PARULA
from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()
from gemseo import LOGGER


class ParallelCoordinates(OptPostProcessor):
    """
    The **ParallelCoordinates** post processing
    builds parallel coordinates plots  among design
    variables, outputs functions and constraints

    x- and y- figure sizes can be changed in option.
    It is possible either to save the plot, to show the plot or both.
    """

    def _run(self, figsize_x=10, figsize_y=2, **options):
        """Visualizes the ScatterPlotMatrix

        :param options: plotting options according to associated json file
        :param figsize_x: X size of the figure Default value = 10
        :param figsize_y: Y size of the figure Default value = 2
        """
        self._plot(figsize_x, figsize_y, **options)

    @staticmethod
    def parallel_coordinates(y_data, x_names, color_criteria, figsize_x, figsize_y):
        """Plots parallel coordinates

        :param y_data: the lines data to plot
        :type y_data: array
        :param x_names: names of the abscissa
        :type x_names: list(str)
        :param color_criteria: the list of  values of same length
                                as y_data to colorize the lines
        :type color_criteria: list(float)
        :param figsize_x: size of figure in horizontal direction (inches)
        :type figsize_x: int
        :param figsize_y: size of figure in vertical direction (inches)
        :type figsize_y: int
        """
        n_x, n_cols = y_data.shape
        assert n_cols == len(x_names)
        assert n_x == len(color_criteria)
        x_values = list(range(n_cols))
        fig = pyplot.figure(figsize=(figsize_x, figsize_y))
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
        self,
        figsize_x,
        figsize_y,
        show=False,
        save=False,
        file_path="para_coord_funcs",
        extension="pdf",
    ):
        """
        Plots the ScatterPlotMatrix graph

        :param figsize_x: size of figure in horizontal direction (inches)
        :type figsize_x: int
        :param figsize_y: size of figure in vertical direction (inches)
        :type figsize_y: int
        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param extension: file extension
        :type extension: str
        """
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
        fig = self.parallel_coordinates(x_vals, x_names, obj_val, figsize_x, figsize_y)
        fig.suptitle(
            "Design variables history colored" + " by '" + obj_name + "'  value"
        )
        root = splitext(file_path)[0]
        self._save_and_show(
            fig,
            save=save,
            show=show,
            file_path=root + "para_coord_des_vars",
            extension=extension,
        )

        func_vals = vals[:, : vals.shape[1] - len(x_names)]
        fig = self.parallel_coordinates(
            func_vals, func_names, obj_val, figsize_x, figsize_y
        )
        fig.suptitle(
            "Objective function and constraints history"
            + " colored by '"
            + obj_name
            + "' value"
        )
        self._save_and_show(
            fig,
            file_path=root + "para_coord_funcs",
            save=save,
            show=show,
            extension=extension,
        )
