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
A scatter plot matrix to display optimization history
*****************************************************
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from matplotlib import pyplot
from pandas.core.frame import DataFrame

try:
    from pandas.tools.plotting import scatter_matrix
except ImportError:
    from pandas.plotting import scatter_matrix

from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()


from gemseo import LOGGER


class ScatterPlotMatrix(OptPostProcessor):
    """
    The **ScatterPlotMatrix** post processing
    builds scatter plot matrix  among design
    variables, outputs functions and constraints.

    The list of variable names has to be passed as arguments
    of the plot method. x- and y- figure sizes can be changed in option.
    It is possible either to save the plot, to show the plot or both.
    """

    def _plot(
        self,
        variables_list,
        figsize_x=10,
        figsize_y=10,
        show=False,
        save=False,
        file_path="scatter_mat",
        extension="pdf",
    ):
        """
        Plots the ScatterPlotMatrix graph

        :param variables_list: the functions names or design variables to plot
        :type variables_list: list(str)
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

        add_dv = False
        all_funcs = self.opt_problem.get_all_functions_names()
        all_dv_names = self.opt_problem.design_space.variables_names
        design_variables = None
        if not variables_list:
            # function list only contains design variables
            vals = self.database.get_x_history()
            vname = self.database.set_dv_names(vals[0].shape[0])
        else:
            design_variables = []
            for func in list(variables_list):
                if func not in all_funcs and func not in all_dv_names:
                    min_f = "-" + func == self.opt_problem.objective.name
                    if min_f and not self.opt_problem.minimize_objective:
                        variables_list[variables_list.index(func)] = "-" + func
                    else:
                        msg = "Cannot build scatter plot matrix,"
                        msg += " Function " + func + " is neither among"
                        msg += " optimization problem functions :"
                        msg += str(all_funcs) + "nor design variables :"
                        msg += str(all_dv_names)
                        raise ValueError(msg)
                if func in self.opt_problem.design_space.variables_names:
                    # if given function is a design variable, then remove it
                    add_dv = True
                    variables_list.remove(func)
                    design_variables.append(func)
            if design_variables == []:
                design_variables = None
            vals, vname, _ = self.database.get_history_array(
                variables_list, design_variables, add_dv=add_dv
            )
        # Next line is a trick for a bug workaround in numpy/matplotlib
        # https://stackoverflow.com/questions/39180873/pandas-dataframe-valueerror-num-must-be-1-num-0-not-1
        vals = (list(x) for x in vals)
        frame = DataFrame(vals, columns=vname)
        scatter_matrix(frame, alpha=1.0, figsize=(figsize_x, figsize_y), diagonal="kde")
        fig = pyplot.gcf()
        fig.tight_layout()
        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )
