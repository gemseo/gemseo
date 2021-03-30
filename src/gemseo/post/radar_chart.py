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
#        :author: Pierre-Jean Barjhoux
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
A radar plot of constraints
***************************
"""
from __future__ import absolute_import, division, unicode_literals

from matplotlib import pyplot
from numpy import concatenate, linspace, max, pi, rad2deg, zeros

from gemseo import LOGGER
from gemseo.post.opt_post_processor import OptPostProcessor


class RadarChart(OptPostProcessor):
    """Plot on radar style chart a list of constraint functions.

    This class has the responsability of plotting on radar style chart a list
    of constraint functions at a given iteration.

    By default, the iteration is the last one.
    It is possible either to save the plot, to show the plot or both.
    """

    def _plot(
        self,
        constraints_list,
        iteration=-1,
        show=False,
        save=False,
        file_path="radar_chart",
        extension="pdf",
        figsize_x=8,
        figsize_y=8,
    ):
        """Plot radar graph.

        :param constraints_list: list of constraints names
        :type constraints_list: list(str)
        :param iteration: number of iteration to post process
        :type iteration: int
        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param extension: file extension
        :type extension: str
        """
        # retrieve the constraints values
        add_dv = False
        all_constr_names = self.opt_problem.get_constraints_names()

        for func in constraints_list:
            if func not in all_constr_names:
                raise ValueError(
                    "Cannot build radar chart,"
                    + " Function "
                    + func
                    + " is not among constraints names"
                    + " or does not exist."
                )

        vals, vname, _ = self.database.get_history_array(
            constraints_list, add_dv=add_dv
        )

        if iteration < -1 or iteration >= len(self.database):
            raise ValueError(
                "iteration should be positive and lower than"
                + " maximum iteration ="
                + str(len(self.database))
            )
        cstr_values = vals[iteration, :]

        # formating values, max and min
        values = cstr_values.ravel()
        vmax = max(cstr_values)
        vmin = min(cstr_values)

        # radar solid grid lines
        pyplot.rc("grid", color="k", linewidth=0.3, linestyle=":")
        pyplot.rc("xtick", labelsize=10)
        pyplot.rc("ytick", labelsize=10)

        # force square figure and square axes looks better for polar, IMO
        figsize = (figsize_x, figsize_y)
        fig = pyplot.figure(figsize=figsize)
        axe = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")
        constr_nb = len(vname)
        constr_trigger = zeros(constr_nb)

        # computes angles
        theta = 2 * pi * linspace(0, 1 - 1.0 / constr_nb, constr_nb)

        # plot lines
        lines_cstr_comp = axe.plot(
            theta, values, color="k", lw=1, label="computed constraints"
        )
        self.__close_line(lines_cstr_comp[0])

        lines_cstr_lim = axe.plot(
            theta,
            constr_trigger,
            color="red",
            ls="--",
            lw=1,
            label="maximum constraint",
        )
        self.__close_line(lines_cstr_lim[0])

        # display settings
        theta_degree = rad2deg(theta)
        axe.set_thetagrids(theta_degree, vname)
        axe.set_rlim([vmin, vmax])
        axe.set_rticks(linspace(vmin, vmax, 6))
        # set title
        if iteration == -1:
            title = "Constraints at last iteration"
        else:
            title = "Constraints at iteration %i" % iteration
        axe.set_title(title)

        # Shrink current axis's height by 10% on the bottom
        box = axe.get_position()
        axe.set_position(
            [box.x0, box.y0 + box.height * 0.15, box.width, box.height * 0.85]
        )

        axe.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))

        # save or show
        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )

    @staticmethod
    def __close_line(line):
        """
        Closes line plotted in the radar chart

        :param : line (matplotlib oject) to close
        """
        x_values, y_values = line.get_data()
        x_values = concatenate((x_values, [x_values[0]]))
        y_values = concatenate((y_values, [y_values[0]]))
        line.set_data(x_values, y_values)
