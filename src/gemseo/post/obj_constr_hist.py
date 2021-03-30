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

import matplotlib.gridspec as gridspec
import numpy as np
from future import standard_library
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import MaxNLocator

from gemseo.post.core.colormaps import PARULA, RG_SEISMIC
from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()
from gemseo import LOGGER


class ObjConstrHist(OptPostProcessor):
    """
    The **ObjConstrHist** post processing
    plots the constraint functions history in lines charts.

    By default, all constraints are considered. A sublist of constraints
    can be passed as options.
    It is possible either to save the plot, to show the plot or both.
    """

    def __init__(self, opt_problem):
        """
        Constructor

        :param opt_problem: the optimization problem to run
        """
        super(ObjConstrHist, self).__init__(opt_problem)
        self.opt_problem = opt_problem
        self.cmap = PARULA  # "viridis"  # "jet"
        self.ineq_cstr_cmap = RG_SEISMIC  # "seismic" "PRGn_r"
        self.eq_cstr_cmap = "seismic"  # "seismic" "PRGn_r"

    def _plot(
        self,
        save=False,
        show=False,
        file_path="obj_constr_hist",
        constr_names=None,
        extension="pdf",
    ):
        """
        Creates the design variables plot

        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param constr_names: names of the constraints to plot
        :type constr_names: list(str)
        :param extension: file extension
        :type extension: str
        """
        obj_name = self.opt_problem.get_objective_name()
        obj_history, x_history, n_iter = self.__get_history(obj_name)
        fmin = np.min(obj_history)
        fmax = np.max(obj_history)
        if not self.opt_problem.minimize_objective and fmax < 0.0:
            obj_history = -obj_history
            fmin = np.min(obj_history)
            fmax = np.max(obj_history)
        grid = gridspec.GridSpec(1, 2, width_ratios=[15, 1], wspace=0.04, hspace=0.6)
        fig = plt.figure(figsize=(11, 6))
        ax1 = fig.add_subplot(grid[0, 0])
        # objective function
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Objective value", fontsize=12)
        plt.plot(np.arange(len(obj_history)), obj_history)
        plt.ylim([fmin, fmax])
        plt.xlim([0, n_iter])
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.title("Evolution of the objective value and maximal constraint ")

        # Set window size
        mng = plt.get_current_fig_manager()
        mng.resize(700, 1000)

        # number of iterations
        nb_iter = x_history.shape[0]

        ineq_vals, eq_vals, ineq_cstr_id, eq_cstr_id = self.__get_constraints(
            constr_names=constr_names
        )

        # concatenate all constraints values (ineq + eq)
        # NB : we take absolute values of equality constraints for color map
        eq_vals_abs = np.abs(eq_vals)
        all_cstr_list_abs = [ineq_vals, eq_vals_abs]
        all_cstr_vals_abs = np.concatenate(
            [cstr for cstr in all_cstr_list_abs if cstr.size > 0], axis=1
        )
        maxi_by_iter = np.amax(all_cstr_vals_abs, axis=1)
        indmaxi_by_iter = np.argmax(all_cstr_vals_abs, axis=1)
        cstr_ids = np.concatenate((ineq_cstr_id, eq_cstr_id))

        values = np.atleast_2d(maxi_by_iter)
        cmap = RG_SEISMIC
        vmax = fmax
        vmin = -vmax
        extent = -0.5, nb_iter - 0.5, fmin, fmax
        im1 = ax1.imshow(
            values,
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
            extent=extent,
            norm=SymLogNorm(linthresh=1.0, vmin=vmin * 0.75, vmax=vmax * 0.75),
            alpha=0.6,
        )

        # add labels with constraint violation information
        all_cstr_list = [ineq_vals, eq_vals]
        all_cstr_vals = np.concatenate(
            [cstr for cstr in all_cstr_list if cstr.size > 0], axis=1
        )
        for iteration, ind in zip(range(nb_iter), indmaxi_by_iter):
            y_vals = 0.5 * (fmax + fmin)
            x_vals = iteration
            max_label = cstr_ids[ind]
            # no abs on equality constraint
            max_value = all_cstr_vals[iteration, ind]
            text = "constraint " + max_label + " = " + str(max_value)
            ax1.text(x_vals - 0.0, y_vals + 1.1 * fmin, text, rotation="vertical")

        # color map
        cax = fig.add_subplot(grid[0, 1])
        thick_min = int(np.log10(1.0))
        thick_max = int(np.log10(np.abs(vmax)))
        thick_num = thick_max - thick_min + 1
        levels_pos = np.logspace(thick_min, thick_max, num=thick_num, base=10.0)
        levels_pos = np.append(levels_pos, vmax)
        levels_neg = np.sort(-levels_pos)
        levels_neg = np.append(levels_neg, 0)
        levels = np.concatenate((levels_neg, levels_pos))

        col_bar = fig.colorbar(im1, cax=cax, ticks=levels)
        col_bar.ax.tick_params(labelsize=9)
        cax.set_xlabel("symlog")

        # save and show
        self._save_and_show(
            fig, save=save, show=show, file_path=file_path, extension=extension
        )

    def __get_history(self, fname):
        """
        Access the optimization history of a function and the design
        variables at which it was computed

        :param fname: name of the function
        :returns: list of function values
            list of design variables values
            list of iterations
        """
        f_hist, x_hist = self.database.get_func_history(fname, x_hist=True)
        f_hist = np.array(f_hist).real
        x_hist = np.array(x_hist).real
        n_iter = x_hist.shape[0]

        return f_hist, x_hist, n_iter

    def __get_constraints(self, constr_names=None):
        """
        Returns constraints with formated shape

        :param constr_name: list of constraint names
        """
        # retrieve the constraints values
        ineq_cstr_names = []
        eq_cstr_names = []
        for cstr in self.opt_problem.get_ineq_constraints():
            if constr_names is None:
                ineq_cstr_names.append(cstr.name)
            else:
                if cstr.name in constr_names:
                    ineq_cstr_names.append(cstr.name)
        for cstr in self.opt_problem.get_eq_constraints():
            if constr_names is None:
                eq_cstr_names.append(cstr.name)
            else:
                if cstr.name in constr_names:
                    eq_cstr_names.append(cstr.name)
        get_hist_array = self.database.get_history_array
        if ineq_cstr_names != []:
            ineq_vals, ineq_id, _ = get_hist_array(ineq_cstr_names, add_dv=False)
        else:
            ineq_vals, ineq_id = np.array([]), np.array([])
        if eq_cstr_names != []:
            eq_vals, eq_cstr_id, _ = get_hist_array(eq_cstr_names, add_dv=False)
        else:
            eq_vals, eq_cstr_id = np.array([]), np.array([])

        # harmonization of tables format because constraints can be vectorial
        # or scalars. *vals.shape[0] = iteration, *vals.shape[1] = cstr values
        ineq_vals = np.atleast_3d(ineq_vals)
        ineq_shape = ineq_vals.shape
        ineq_vals = np.reshape(
            ineq_vals, (ineq_shape[0], ineq_shape[1] * ineq_shape[2])
        )
        eq_vals = np.atleast_3d(eq_vals)
        eq_shape = eq_vals.shape
        eq_vals = np.reshape(eq_vals, (eq_shape[0], eq_shape[1] * eq_shape[2]))

        return ineq_vals, eq_vals, ineq_id, eq_cstr_id
