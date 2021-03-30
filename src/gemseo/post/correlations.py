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
Correlations in the optimization database
*****************************************
"""

from __future__ import absolute_import, division, unicode_literals

from os.path import basename, dirname, join, splitext

import matplotlib.gridspec as gridspec
import numpy as np
import pylab
from future import standard_library
from matplotlib import ticker
from numpy import atleast_2d

from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()


from gemseo import LOGGER


class Correlations(OptPostProcessor):
    """
    The **Correlations** post processing
    builds scatter plots of correlated variables among design
    variables, outputs functions and constraints

    The plot method considers all variable correlations
    greater than 95%. An other level value, a sublist of variable names
    or both can be passed as options. The x- and y- figure sizes
    can also be modified in option.
    It is possible either to save the plot, to show the plot or both.
    """

    def _run(self, **options):
        """Visualizes the optimization history

        :param options: options for the post processing,
            see associated JSON file
        """
        functions = self.opt_problem.get_all_functions()
        func_names = [func.name for func in functions]
        self._plot(func_names, **options)

    def _plot(
        self,
        func_names=None,
        coeff_limit=0.95,
        n_plots_x=5,
        n_plots_y=5,
        save=False,
        show=False,
        file_path=None,
        extension="pdf",
    ):
        """
        Plots the correlations graph

        :param coeff_limit: if the correlation between the variables
            is lower than coeff_limit, the plot is not made
        :type coeff_limit: bool
        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param func_names: the func_names on which correlations is computed
        :type func_names: list(str)
        :param n_plots_x: number of horizontal plots
        :type n_plots_x: int
        :param n_plots_y: number of vertical plots
        :type n_plots_y: int
        :param extension: file extension
        :type extension: str
        """
        n_slide = 0
        values_array, variables_names, _ = self.database.get_history_array(
            func_names, None, True, 0.0
        )
        corr_coeffs_array = self.__compute_correlations(values_array)
        i_corr, j_corr = np.where(
            (np.abs(corr_coeffs_array) > coeff_limit)
            & (np.abs(corr_coeffs_array) < (1.0 - 1e-9))
        )
        LOGGER.info("Detected %s correlations > %s", i_corr.size, coeff_limit)
        if i_corr.size <= 16:
            n_plots_x = 4
            n_plots_y = 4
        figs = []
        spec = gridspec.GridSpec(n_plots_y, n_plots_x, wspace=0.3, hspace=0.75)
        spec.update(top=0.95, bottom=0.06, left=0.08, right=0.95)
        fig = None
        fig_indx = 0
        if file_path is not None:
            root = splitext(file_path)[0]
            root_dir = dirname(root)
            base_n = basename(root)
        else:
            root_dir = "."
            base_n = ""
        for plot_index, (i, j) in enumerate(zip(i_corr, j_corr)):
            plot_index_loc = plot_index % (n_plots_x * n_plots_y)
            if plot_index_loc == 0:
                if fig is not None:  # Save previous plot
                    fig_indx += 1
                    base_loc = base_n + "correlations_" + str(fig_indx)
                    fpath = join(root_dir, base_loc)
                    self._save_and_show(
                        fig, file_path=fpath, save=save, show=show, extension=extension
                    )
                    pylab.plt.close(fig)
                fig = pylab.plt.figure()
                figs.append(fig)
                mng = pylab.plt.get_current_fig_manager()
                mng.resize(1200, 900)
                ticker.MaxNLocator(nbins=3)

            # plt.suptitle('All variables are normalized')
            self.__create_sub_correlation_plot(
                i,
                j,
                corr_coeffs_array[i, j],
                fig,
                spec,
                plot_index_loc,
                n_plots_y,
                n_plots_x,
                values_array,
                variables_names,
            )
        if fig is not None:
            base_loc = base_n + "correlations_" + str(fig_indx + 1)
            fpath = join(root_dir, base_loc)
            self._save_and_show(
                fig, save=save, show=show, file_path=fpath, extension=extension
            )
            pylab.plt.close(fig)
        return n_slide

    def __create_sub_correlation_plot(
        self,
        i_ind,
        j_ind,
        corr_coeff,
        fig,
        spec,
        plot_index,
        n_plot_v,
        n_plot_h,
        values_array,
        variables_names,
    ):
        """Creates a correlation plot"""
        gs_curr = spec[int(plot_index / n_plot_v), plot_index % n_plot_h]
        ax1 = fig.add_subplot(gs_curr)
        x_plt = values_array[:, i_ind]
        y_plt = values_array[:, j_ind]
        ax1.scatter(x_plt, y_plt, c="b", s=30)
        self.out_data_dict[(i_ind, j_ind)] = (
            variables_names[i_ind],
            variables_names[j_ind],
            corr_coeff,
        )
        ax1.set_xlabel(variables_names[i_ind], fontsize=9)
        # Update y labels spacing
        start, stop = ax1.get_ylim()
        ax1.yaxis.set_ticks(np.arange(start, stop, 0.24999999 * (stop - start)))
        start, stop = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(start, stop, 0.24999999 * (stop - start)))
        ax1.set_ylabel(variables_names[j_ind], fontsize=10)
        ax1.tick_params(labelsize=10)
        ax1.set_title("R=%5f" % corr_coeff, fontsize=12)
        ax1.grid()

    @classmethod
    def __compute_correlations(cls, values_array):
        """Compute correlations"""
        ccoeff = np.corrcoef(values_array.astype(float), rowvar=False)
        return np.tril(atleast_2d(ccoeff))  # Keep upper diagonal only
