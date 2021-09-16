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
"""Correlations in the optimization database."""

from __future__ import division, unicode_literals

import logging
from functools import partial
from typing import List, Optional, Sequence, Tuple

import matplotlib.gridspec as gridspec
import numpy as np
import pylab
from matplotlib import ticker
from matplotlib.figure import Figure
from numpy import atleast_2d, ndarray

from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.py23_compat import fullmatch

LOGGER = logging.getLogger(__name__)


class Correlations(OptPostProcessor):
    """Scatter plots of the correlated variables.

    These variables can be design variables, outputs functions or constraints.

    The plot method considers all the variable correlations greater than 95%.
    Another level value, a sublist of variable names or both can be passed as options.
    """

    DEFAULT_FIG_SIZE = (15.0, 10.0)

    def _plot(
        self,
        func_names=None,  # type: Optional[Sequence[str]]
        coeff_limit=0.95,  # type: float
        n_plots_x=5,  # type: int
        n_plots_y=5,  # type: int
    ):  # type: (...) -> None
        """
        Args:
            func_names: The function names subset
                for which the correlations are computed.
                If None, all functions are considered.
            coeff_limit: The plot is not made
                if the correlation between the variables is lower than this limit.
            n_plots_x: The number of horizontal plots.
            n_plots_y: The number of vertical plots.

        Raises:
            ValueError: If an element of `func_names` is not a function
                defined in `opt_problem`.
        """
        functions = self.opt_problem.get_all_functions()
        all_func_names = [func.name for func in functions]

        if not func_names:
            func_names = all_func_names
        elif set(func_names).issubset(all_func_names):
            func_names = [
                func_name for func_name in all_func_names if func_name in func_names
            ]
        else:
            raise ValueError(
                "The following elements are not "
                "functions: {}. Defined functions are {}.".format(
                    ", ".join(set(func_names) - set(all_func_names)),
                    ", ".join(all_func_names),
                )
            )

        values_array, variables_names, _ = self.database.get_history_array(
            func_names, None, True, 0.0
        )

        variables_names = self.__sort_variables_names(variables_names, func_names)

        corr_coeffs_array = self.__compute_correlations(values_array)
        i_corr, j_corr = np.where(
            (np.abs(corr_coeffs_array) > coeff_limit)
            & (np.abs(corr_coeffs_array) < (1.0 - 1e-9))
        )
        LOGGER.info("Detected %s correlations > %s", i_corr.size, coeff_limit)

        if i_corr.size <= 16:
            n_plots_x = 4
            n_plots_y = 4
        spec = gridspec.GridSpec(n_plots_y, n_plots_x, wspace=0.3, hspace=0.75)
        spec.update(top=0.95, bottom=0.06, left=0.08, right=0.95)
        fig = None
        fig_indx = 0
        for plot_index, (i, j) in enumerate(zip(i_corr, j_corr)):
            plot_index_loc = plot_index % (n_plots_x * n_plots_y)
            if plot_index_loc == 0:
                if fig is not None:  # Save previous plot
                    fig_indx += 1
                    self._add_figure(fig)
                fig = pylab.plt.figure(figsize=self.DEFAULT_FIG_SIZE)
                mng = pylab.plt.get_current_fig_manager()
                mng.resize(1200, 900)
                ticker.MaxNLocator(nbins=3)

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
            self._add_figure(fig)

    def __create_sub_correlation_plot(
        self,
        i_ind,  # type: int
        j_ind,  # type: int
        corr_coeff,  # type: ndarray
        fig,  # type: Figure
        spec,  # type: gridspec
        plot_index,  # type: int
        n_plot_v,  # type: int
        n_plot_h,  # type: int
        values_array,  # type: ndarray
        variables_names,  # type: Sequence[str]
    ):  # type: (...)-> None
        """Create a correlation plot.

        Args:
            i_ind: The index for the x-axis data.
            j_ind: The index for the y-axis data.
            corr_coeff: The correlation coefficients.
            fig: The figure where the subplot will be placed.
            spec: The matplotlib grid structure.
            plot_index: The local plot index.
            n_plot_v: The number of vertical plots.
            n_plot_h: The number of horizontal plots.
            values_array: The function values from the optimization history.
            variables_names: The variables names.
        """
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
        ax1.set_title("R={:.5f}".format(corr_coeff), fontsize=12)
        ax1.grid()

    @classmethod
    def __compute_correlations(
        cls, values_array  # type: ndarray
    ):  # type: (...)-> ndarray
        """Compute correlations.

        Args:
            values_array: The values to compute the correlations.

        Returns:
            The lower diagonal of the correlations matrix.
        """
        ccoeff = np.corrcoef(values_array.astype(float), rowvar=False)
        return np.tril(atleast_2d(ccoeff))  # Keep lower diagonal only

    def __sort_variables_names(
        self,
        variables_names,  # type: Sequence[str]
        func_names,  # type: Sequence[str]
    ):  # type: (...)-> List[str]
        """Sort the expanded variable names using func_names as the pattern.

        In addition to sorting the expanded variable names, this method
        replaces the default hard-coded vectors (x_1, x_2, ... x_n) with
        the names given by the user.

        Args:
            variables_names: The expanded variable names to be sorted.
            func_names: The functions names in the required order.

        Returns:
            The sorted expanded variable names.
        """
        variables_names.sort(key=partial(self.func_order, func_names))
        x_names = self._generate_x_names()

        return variables_names[: -len(x_names)] + x_names

    @staticmethod
    def func_order(
        func_names,  # type: Sequence[str]
        x,  # type: str
    ):  # type: (...) -> Tuple[int, str]
        """Key function to sort function components.

        Args:
            func_names: The functions names in the required order.
            x: An element from a list.

        Returns:
            The index to be given to the sort method and the
                function name associated to that index.
        """

        for i, func_name in enumerate(func_names):
            if fullmatch(r"{}(_\d+)?".format(func_name), x):
                return (i, x.replace(func_name, ""))

        return (len(func_names) + 1, x)
