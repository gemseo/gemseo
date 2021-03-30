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
Display a Pareto Front
**********************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

from gemseo.algos.pareto_front import generate_pareto_plots
from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()


from gemseo import LOGGER


class ParetoFront(OptPostProcessor):
    """
    Compute the Pareto front
    Search for all non dominated points, ie there exists j such that
    there is no lower value for obj_values[:,j] that does not degrade
    at least one other objective  obj_values[:,i].

    Generates a plot or a matrix of plots if there are more than 2 objectives.
    Plots in red the locally non dominated points
    for the currrent two objectives.
    Plot in green the globally (all objectives) Pareto optimal points.
    """

    def _plot(
        self,
        objectives=None,
        objectives_labels=None,
        figsize_x=10,
        figsize_y=10,
        show=False,
        save=False,
        file_path=None,
        extension="pdf",
    ):
        """
        Plots the Pareto front

        :param objectives: the functions names or design variables to plot
            if None, use the objective function (may be a vector)
        :type objectives: list(str)
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
        if objectives is None:
            objectives = [self.opt_problem.objective.name]
        add_dv = False
        all_funcs = self.opt_problem.get_all_functions_names()
        all_dv_names = self.opt_problem.design_space.variables_names
        design_variables = None
        if not objectives:
            # function list only contains design variables
            vals = self.database.get_x_history()
            vname = self.database.set_dv_names(vals[0].shape[0])
        else:
            design_variables = []
            for func in list(objectives):
                if func not in all_funcs and func not in all_dv_names:
                    min_f = "-" + func == self.opt_problem.objective.name
                    if min_f and not self.opt_problem.minimize_objective:
                        objectives[objectives.index(func)] = "-" + func
                    else:
                        msg = "Cannot build Pareto front,"
                        msg += " Function " + func + " is neither among"
                        msg += " optimization problem functions :"
                        msg += str(all_funcs) + "nor design variables :"
                        msg += str(all_dv_names)
                        raise ValueError(msg)
                if func in self.opt_problem.design_space.variables_names:
                    # if given function is a design variable, then remove it
                    add_dv = True
                    objectives.remove(func)
                    design_variables.append(func)
            if design_variables == []:
                design_variables = None
            vals, vname, _ = self.database.get_history_array(
                objectives, design_variables, add_dv=add_dv
            )
        if objectives_labels is not None:
            assert len(vname) == len(objectives_labels)
            vname = objectives_labels
        fig = generate_pareto_plots(vals, vname, figsize=(figsize_x, figsize_y))

        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )
