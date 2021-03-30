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
A constraints plot
******************
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from matplotlib import pyplot as plt
from numpy import arange, array

from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()


from gemseo import LOGGER


class BasicHistory(OptPostProcessor):
    """
    The **BasicHistory** post processing plots any of the constraint
    or objective functions
    w.r.t. optimization iterations or sampling snapshots.

    The plot method requires the list of variable names to plot.
    It is possible either to save the plot, to show the plot or both.
    """

    def _plot(
        self,
        data_list,
        show=False,
        save=False,
        file_path="basic_history",
        extension="pdf",
    ):
        """
        Plots the optimization history:
        1 plot for the constraints

        :param data_list: list of variable names
        :type data_list: list(str)
        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param variables_list: list of the constraints (func name)
        :type variables_list: list(str)
        :param extension: file extension
        :type extension: str
        """
        fig = plt.figure(figsize=(11, 6))
        plt.xlabel("Iterations", fontsize=12)
        dspace = self.opt_problem.design_space
        desvars = dspace.variables_names

        has_dv = False
        for data_name in data_list:
            if data_name in desvars:
                has_dv = True
                break
        dv_hist = None
        if has_dv:
            dv_hist = array(self.database.get_x_history())
        for data_name in data_list:
            if data_name in desvars:
                mask = []
                for currvar in dspace.variables_names:
                    size = dspace.variables_sizes[currvar]
                    isvar = data_name == currvar
                    mask += [isvar] * size

                data_hist = dv_hist[:, mask]
            else:
                data_hist = self.database.get_func_history(data_name, x_hist=False)
            labels = data_name

            data_hist = array(data_hist).real
            plt.plot(arange(len(data_hist)), data_hist, label=labels)

        plt.title("History plot")
        plt.legend()
        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )
