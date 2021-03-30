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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""
Basic display of optimization history : functions, and x
********************************************************
"""
from __future__ import absolute_import, division, unicode_literals

import matplotlib.gridspec as gridspec
import pylab
from future import standard_library
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatter, MaxNLocator
from numpy import abs as np_abs
from numpy import append, arange, argmin, array, atleast_2d, concatenate, hstack, isnan
from numpy import log10 as np_log10
from numpy import logspace
from numpy import max as np_max
from numpy import min as np_min
from numpy import ones, ones_like
from numpy import sort as np_sort
from numpy import vstack, where
from numpy.linalg import norm

from gemseo.core.function import MDOFunction
from gemseo.post.core.colormaps import PARULA, RG_SEISMIC
from gemseo.post.core.hessians import SR1Approx
from gemseo.post.opt_post_processor import OptPostProcessor

standard_library.install_aliases()
from gemseo import LOGGER


class OptHistoryView(OptPostProcessor):
    """
    The **OptHistoryView** post processing
    performs separated plots:
    the design variables history,
    the objective function history,
    the history of hessian approximation of the objective,
    the inequality constraint history,
    the equality constraint history,
    and constraints histories.

    By default, all design variables are considered.
    A sublist of design variables can be passed as options.
    Minimum and maximum values for the plot can be passed as options.
    The objective function can also be represented in terms of difference
    w.r.t. the initial value
    It is possible either to save the plot, to show the plot or both.
    """

    def __init__(self, opt_problem):
        """
        Constructor

        :param opt_problem: the optimization problem to run
        """
        super(OptHistoryView, self).__init__(opt_problem)
        self.cmap = PARULA  # "viridis"  # "jet"
        self.ineq_cstr_cmap = RG_SEISMIC  # "seismic" "PRGn_r"
        self.eq_cstr_cmap = "seismic"  # "seismic" "PRGn_r"

    def _plot(
        self,
        show=False,
        save=False,
        file_path=None,
        variables_names=None,
        obj_min=None,
        obj_max=None,
        obj_relative=False,
        extension="pdf",
    ):
        """
        Plots the optimization history:
        1 plot for the design variables
        1 plot for the objective function
        1 plot for the Hessian approximation of the objective
        1 plot for inequality constraints
        1 plot for equality constraints

        :param show: if True, displays the plot windows
        :type show: bool
        :param save: if True, exports plot to pdf
        :type save: bool
        :param file_path: the base paths of the files to export
        :type file_path: str
        :param variables_names: list of the names of the variables to display
        :type variables_names: list(str)
        :param obj_max: maximum value for the objective in the plot
        :type obj_max: float
        :param obj_min: minimum value for the objective in the plot
        :type obj_min: float
        :param obj_relative: plot the objective value difference
            with the initial value
        :type obj_relative: bool
        :param extension: file extension
        :type extension: str
        """
        # compute the names of the inequality and equality constraints
        ineq_cstr = self.opt_problem.get_ineq_constraints()
        ineq_cstr_names = [c.name for c in ineq_cstr]
        eq_cstr = self.opt_problem.get_eq_constraints()
        eq_cstr_names = [c.name for c in eq_cstr]

        obj_name = self.opt_problem.get_objective_name()
        obj_history, x_history, n_iter = self._get_history(obj_name, variables_names)

        # design variables
        self._create_variables_plot(
            x_history, variables_names, save, show, file_path, extension
        )

        # objective function
        self._create_obj_plot(
            obj_history,
            n_iter,
            save,
            show,
            file_path,
            obj_min,
            obj_max,
            obj_relative=obj_relative,
            extension=extension,
        )

        self._create_x_star_plot(
            x_history, n_iter, save, show, file_path=file_path, extension=extension
        )

        # Hessian plot
        plot_hessian = not self.database.is_func_grad_history_empty(obj_name)
        if plot_hessian:
            self._create_hessian_approx_plot(
                self.database, obj_name, save, show, file_path, extension
            )

        # inequality and equality constraints
        self._plot_cstr_history(
            ineq_cstr_names, MDOFunction.TYPE_INEQ, save, show, file_path, extension
        )
        self._plot_cstr_history(
            eq_cstr_names, MDOFunction.TYPE_EQ, save, show, file_path, extension
        )

    def _plot_cstr_history(
        self, cstr_names, cstr_type, save, show, file_path, extension
    ):
        """
        Create the plot for (in)equality constraints

        :param cstr_names: names of the constraints
        :param cstr_type: type of the constraints in {MDOFunction.TYPE_INEQ,
            MDOFunction.TYPE_EQ}
        :param show: if True, displays the plot windows
        :param save: if True, exports plot to pdf
        :param file_path: the base paths of the files to export
        :param extension: file extension
        """
        if cstr_names is not None:
            _, constraints_history = self._get_constraints(cstr_names)
            self._create_cstr_plot(
                constraints_history,
                cstr_type,
                cstr_names,
                save,
                show,
                file_path,
                extension,
            )

    def _get_history(self, fname, variables_names):
        """
        Access the optimization history of a function and the design
        variables at which it was computed

        :param fname: name of the function
        :param variables_names: list of the names of the variables to display
        :returns: list of function values
            list of design variables values
            list of iterations
        """
        f_hist, x_hist = self.database.get_func_history(fname, x_hist=True)
        f_hist = array(f_hist).real
        x_hist = array(x_hist).real

        if variables_names is not None:
            # select only the interesting columns
            block_list = []
            column = 0
            for var in self.opt_problem.design_space.variables_names:
                if var in variables_names:
                    size = self.opt_problem.design_space.variables_sizes[var]
                    block_list.append(x_hist[:, column : column + size])
                    column += size
            # concatenate the blocks
            x_hist = hstack(block_list)
        n_iter = x_hist.shape[0]
        return f_hist, x_hist, n_iter

    def _get_constraints(self, cstr_names):
        """
        Extracts a history of constraints

        :param cstr_names: names of the constraints
        :returns: bounds of the constraints and history array
        """
        available_data_names = self.database.get_all_data_names()
        for cstr_name in cstr_names:
            if cstr_name not in available_data_names:
                cstr_names.remove(cstr_name)

        constraints_history = []
        bnd_list = -1e300 * ones(len(cstr_names))

        for cstr_i, cstr_name in enumerate(cstr_names):

            cstr_history = array(self.database.get_func_history(cstr_name)).real

            constraints_history.append(cstr_history)
            bnd_list[cstr_i] = max(
                bnd_list[cstr_i],
                max(abs(np_min(cstr_history)), abs(np_max(cstr_history))),
            )

        return bnd_list, constraints_history

    def _normalize_x_hist(self, x_history, variables_names):
        """
        Normalizes the design variables history

        :param x_history: the history for the design variables
        :param variables_names: list of the names of the variables to display
        :returns: the normalized design variables array
        """
        x_hist_n = x_history.copy()
        lower_bounds = self.opt_problem.design_space.get_lower_bounds(variables_names)
        upper_bounds = self.opt_problem.design_space.get_upper_bounds(variables_names)
        norm_coeff = 1 / (np_abs(upper_bounds - lower_bounds))
        for i in range(x_history.shape[0]):
            x_hist_n[i, :] = (x_hist_n[i, :] - lower_bounds) * norm_coeff
        return x_hist_n

    def _create_variables_plot(
        self, x_history, variables_names, save, show, file_path=None, extension="pdf"
    ):
        """
        Creates the design variables plot

        :param x_history: the design variables history
        :param variables_names: list of the names of the variables to display
        :param n_variables: number of parameters
        :param save: saves the plot to a file
        :param show: shows the matplotlib figure
        :param file_path: the base paths of the files to export
        :param extension: file extension
        """
        if len(x_history) < 2:
            return
        n_variables = x_history.shape[1]
        norm_x_history = self._normalize_x_hist(x_history, variables_names)

        fig = pylab.plt.figure(figsize=(11, 6))
        grid = gridspec.GridSpec(1, 2, width_ratios=[15, 1], wspace=0.04, hspace=0.6)

        # design variables
        ax1 = fig.add_subplot(grid[0, 0])
        im1 = ax1.imshow(
            norm_x_history.T,
            cmap=self.cmap,
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        # y ticks: names of the variables
        design_space = self.opt_problem.design_space
        y_labels = []
        if variables_names is None:
            variables_names = design_space.variables_names
        for variable_name in variables_names:
            size = design_space.variables_sizes[variable_name]
            name = variable_name
            if size > 1:
                name += " (0)"
            y_labels.append(name)
            for i in range(1, size):
                y_labels.append("(" + str(i) + ")")

        ax1.set_yticklabels(y_labels)
        ax1.set_yticks(arange(n_variables))
        # ax1.invert_yaxis()

        ax1.set_title("Evolution of the optimization variables")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # colorbar
        ax2 = fig.add_subplot(grid[0, 1])
        fig.colorbar(im1, cax=ax2)

        if save:
            if file_path is not None:
                path = file_path + "_variables_history." + extension
            else:
                path = "design_variables_history." + extension
            pylab.plt.savefig(path)
            self.output_files.append(path)

        # Set window size
        mng = pylab.plt.get_current_fig_manager()
        mng.resize(700, 1000)

        if show:
            pylab.plt.show()
        else:
            pylab.plt.close(fig)

    def _create_obj_plot(
        self,
        obj_history,
        n_iter,
        save,
        show,
        file_path=None,
        obj_min=None,
        obj_max=None,
        obj_relative=False,
        extension="pdf",
    ):
        """
        Creates the design variables plot

        :param obj_history: history of the objective function
        :param n_iter: number of iterations
        :param save: saves the plot to the disc
        :param show: shows the matplotlib figures
        :param file_path: the base paths of the files to export
        :param obj_max: maximum value for the objective in the plot
        :param obj_min: minimum value for the objective in the plot
        :param obj_relative: plot the objective value difference
            with the initial value
        :param extension: file extension
        """
        # if max problem, plot -objective value
        if not self.opt_problem.minimize_objective:
            obj_history = -obj_history
        if obj_relative:
            LOGGER.info(
                "Plot of optimization history "
                "with relative variation compared to "
                "initial point objective value = %s",
                obj_history[0],
            )
            obj_history -= obj_history[0]
        # Remove nans
        n_x = len(obj_history)
        x_absc = arange(n_x)
        x_absc_nan = None
        idx_nan = isnan(obj_history)

        if idx_nan.size > 0:
            obj_history = obj_history[~idx_nan]
            x_absc_nan = x_absc[idx_nan]
            x_absc = x_absc[~idx_nan]

        fmin = np_min(obj_history)
        fmax = np_max(obj_history)

        fig = pylab.plt.figure(figsize=(11, 6))
        # objective function
        pylab.plt.xlabel("Iterations", fontsize=12)
        pylab.plt.ylabel("Objective value", fontsize=12)

        pylab.plt.plot(x_absc, obj_history)

        if idx_nan.size > 0:
            for x_i in x_absc_nan:
                pylab.plt.axvline(x_i, color="purple")

        if obj_min is not None and obj_min < fmin:
            fmin = obj_min
        if obj_max is not None and obj_max > fmax:
            fmax = obj_max
        pylab.plt.ylim([fmin, fmax])
        pylab.plt.xlim([0, n_iter])
        ax1 = fig.gca()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        pylab.plt.grid(True)
        pylab.plt.title("Evolution of the objective value")

        if save:
            if file_path is not None:
                path = file_path + "_obj_history." + extension
            else:
                path = "objective_function_history." + extension
            pylab.plt.savefig(path)
            self.output_files.append(path)

        # Set window size
        mng = pylab.plt.get_current_fig_manager()
        mng.resize(700, 1000)

        if show:
            pylab.plt.show()
        else:
            pylab.plt.close(fig)

    def _create_x_star_plot(
        self, x_history, n_iter, save, show, file_path=None, extension="pdf"
    ):
        """
        Creates the design variables plot

        :param x_history: history of the design variables
        :param x_opt: x optimum
        :param n_iter: number of iterations
        :param save: save the figure
        :param show: show the figure
        :param file_path: the base paths of the files to export
        :param extension: file extension
        """
        fig = pylab.plt.figure(figsize=(11, 6))
        # objective function
        pylab.plt.xlabel("Iterations", fontsize=12)
        pylab.plt.ylabel("||x-x*||", fontsize=12)

        n_i = x_history.shape[0]
        _, x_opt, _, _, _ = self.opt_problem.get_optimum()

        normalize = self.opt_problem.design_space.normalize_vect

        x_xstar = [
            norm(normalize(x_history[i, :]) - normalize(x_opt)) for i in range(n_i)
        ]
        # Draw a vertical line at the optimum
        ind_opt = argmin(x_xstar)
        pylab.plt.axvline(x=ind_opt, color="r")
        pylab.plt.semilogy(arange(len(x_xstar)), x_xstar)
        # ======================================================================
        # try:
        #     pylab.plt.semilogy(np.arange(len(x_xstar)), x_xstar)
        # except ValueError:
        #     LOGGER.warning("Cannot use log scale for x_star plot since" +
        #                    "all values are not positive !")
        # ======================================================================
        ax1 = fig.gca()
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        pylab.plt.grid(True)
        pylab.plt.title("Distance to the optimum")
        pylab.plt.xlim([0, n_iter])

        if save:
            if file_path is not None:
                path = file_path + "_x_xstar_history." + extension
            else:
                path = "x_xstar_history." + extension
            pylab.plt.savefig(path)
            self.output_files.append(path)

        # Set window size
        mng = pylab.plt.get_current_fig_manager()
        mng.resize(700, 1000)

        if show:
            pylab.plt.show()
        else:
            pylab.plt.close(fig)

    @staticmethod
    def _cstr_number(cstr_names, cstr_history):
        """
        Computes the total scalar constraints number

        :param cstr_names: names of the constraints
        :param cstr_history: history of the constraints
        :returns: number of constraints
        """
        n_cstr = 0
        for cstr_i in range(len(cstr_names)):
            c_hist_loc = atleast_2d(cstr_history[cstr_i]).T
            if c_hist_loc.shape[1] == 1:
                c_hist_loc = c_hist_loc.T
            n_cstr += c_hist_loc.shape[0]
        LOGGER.debug("Total constraints number =%s", n_cstr)
        return n_cstr

    def _create_cstr_plot(
        self,
        cstr_history,
        cstr_type,
        cstr_names,
        save,
        show,
        file_path=None,
        extension="pdf",
    ):
        """
        Creates the constraints plot: 1 line per constraint component

        :param cstr_history: history of the constraints
        :param cstr_names: names of the constraints
        :param save: if True, exports plot to pdf
        :returns: the plot
        :param file_path: the base paths of the files to export
        :param extension: file extension
        """
        n_cstr = self._cstr_number(cstr_names, cstr_history)
        if n_cstr == 0:
            return

        # matrix of all constraints' values
        cstr_matrix = None
        vmax = 0.0
        cstr_labels = []

        max_iter = 0
        for (i, cstr_history_i) in enumerate(cstr_history):
            history_i = atleast_2d(cstr_history_i).T
            if history_i.shape[1] == 1:
                history_i = history_i.T

            if max_iter < history_i.shape[1]:
                max_iter = history_i.shape[1]

        for (i, cstr_history_i) in enumerate(cstr_history):
            history_i = atleast_2d(cstr_history_i).T
            if history_i.shape[1] == 1:
                history_i = history_i.T

            nb_components = history_i.shape[0]

            if history_i.shape[1] == max_iter:  # TEST
                for component_j in range(nb_components):
                    # compute the label of the constraint
                    if component_j == 0:
                        cstr_name = cstr_names[i]
                        if nb_components >= 2:
                            cstr_name += " (" + str(component_j + 1) + ")"
                        cstr_labels.append(cstr_name)
                    else:
                        cstr_labels.append("(" + str(component_j + 1) + ")")

                    history_i_j = atleast_2d(history_i[component_j, :])

                    # max value
                    notnans = ~isnan(history_i_j)
                    vmax = max(vmax, np_max(np_abs(history_i_j[notnans])))

                    # build the constraint matrix
                    if cstr_matrix is None:
                        cstr_matrix = history_i_j
                    else:
                        cstr_matrix = vstack((cstr_matrix, history_i_j))

        fig = self._build_cstr_fig(cstr_matrix, cstr_type, vmax, n_cstr, cstr_labels)

        if save:
            if file_path is not None:
                path = file_path + "_" + cstr_type + "_constraints_history." + extension
            else:
                path = cstr_type + "_constraints_history." + extension
            pylab.plt.savefig(path)
            self.output_files.append(path)
        if show:
            pylab.plt.show()
        else:
            pylab.plt.close(fig)

    def _build_cstr_fig(self, cstr_matrix, cstr_type, vmax, n_cstr, cstr_labels):
        """
        Builds the constraints figure

        :param cstr_matrix : matrix of constraints values
        :param cstr_labels: labels for the constraints
        :param vmax: maximum constraint absolute value
        :param n_cstr: number of constraints
        :param cstr_labels : labels of constraints names
        :returns: the matplotlib figure
        """
        # cmap of the constraints
        fullname = "equality"
        if cstr_type == MDOFunction.TYPE_EQ:
            cmap = self.eq_cstr_cmap
        else:
            cmap = self.ineq_cstr_cmap
            fullname = "in" + fullname

        idx_nan = isnan(cstr_matrix)
        hasnan = idx_nan.any()
        if hasnan > 0:
            cstr_matrix[idx_nan] = 0.0

        # generation of the image
        fig = pylab.plt.figure(figsize=(11, 6))
        grid = gridspec.GridSpec(1, 2, width_ratios=[15, 1], wspace=0.04, hspace=0.6)
        ax1 = fig.add_subplot(grid[0, 0])
        im1 = ax1.imshow(
            cstr_matrix,
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
            norm=SymLogNorm(linthresh=1.0, vmin=-vmax, vmax=vmax),
        )
        if hasnan > 0:
            x_absc_nan = where(idx_nan.any(axis=0))[0]
            for x_i in x_absc_nan:
                pylab.plt.axvline(x_i, color="purple")

        ax1.tick_params(labelsize=9)
        ax1.set_yticks(list(range(n_cstr)))
        ax1.set_yticklabels(cstr_labels)

        ax1.set_xlabel("evaluations")
        ax1.set_title("Evolution of the " + fullname + " constraints")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax1.hlines(
            list(range(len(cstr_matrix))),
            [-0.5],
            [len(cstr_matrix[0]) - 0.5],
            alpha=0.1,
            lw=0.5,
        )

        # color map
        cax = fig.add_subplot(grid[0, 1])
        if 0.0 < vmax < 1.0:
            thick_min = int(np_log10(vmax))
        else:
            thick_min = 0
        if vmax > 1.0:
            thick_max = int(np_log10(vmax))
        else:
            thick_max = 0
        thick_num = thick_max - thick_min + 1
        levels_pos = logspace(thick_min, thick_max, num=thick_num, base=10.0)
        if vmax != 0.0:
            levels_pos = np_sort(append(levels_pos, vmax))
        levels_neg = np_sort(-levels_pos)
        levels_neg = append(levels_neg, 0)
        levels = concatenate((levels_neg, levels_pos))
        col_bar = fig.colorbar(im1, cax=cax, ticks=levels)
        col_bar.ax.tick_params(labelsize=9)
        cax.set_xlabel("symlog")

        mng = pylab.plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        mng.resize(700, 1000)
        return fig

    def _create_hessian_approx_plot(
        self, history, obj_name, save, show, file_path=None, extension="pdf"
    ):
        """
        Creates the plot of the Hessian approximation

        :param history: the optimization history
        :param obj_name: the function name
        :param extension: file extension
        """
        try:
            approximator = SR1Approx(history)
            _, diag, _, _ = approximator.build_approximation(
                funcname=obj_name, save_diag=True
            )
            diag = [ones_like(diag[0])] + diag  # Add first iteration blank
            diag = array(diag).T
        except ValueError as err:
            LOGGER.warning("Failed to create Hessian approximation: %s", str(err))
            return

        # if max problem, plot -Hessian
        if not self.opt_problem.minimize_objective:
            diag = -diag

        fig = pylab.plt.figure(figsize=(11, 6))
        grid = gridspec.GridSpec(1, 2, width_ratios=[15, 1], wspace=0.04, hspace=0.6)
        # matrix
        axe = fig.add_subplot(grid[0, 0])

        axe.set_title("Hessian diagonal approximation")
        axe.set_xlabel("Iterations", fontsize=12)
        axe.set_ylabel("Variable id", fontsize=12)
        axe.xaxis.set_major_locator(MaxNLocator(integer=True))
        vmax = max(abs(np_max(diag)), abs(np_min(diag)))
        linthresh = 10 ** ((np_log10(vmax) - 5.0))
        img = axe.imshow(
            diag.real,
            cmap=self.cmap,
            interpolation="nearest",
            aspect="auto",
            norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax),
        )
        axe.invert_yaxis()

        # colorbar
        vmax = max(abs(np_max(diag)), abs(np_min(diag)))
        thick_min = int(np_log10(linthresh))
        thick_max = int(np_log10(vmax))
        thick_num = thick_max - thick_min + 1
        levels_pos = logspace(thick_min, thick_max, num=thick_num, base=10.0)
        levels_pos = append(levels_pos, vmax)
        levels_neg = np_sort(-levels_pos)
        levels_neg = append(levels_neg, 0)
        levels = concatenate((levels_neg, levels_pos))

        l_f = LogFormatter(base=10, labelOnlyBase=False)
        cax = fig.add_subplot(grid[0, 1])
        cax.invert_yaxis()
        fig.colorbar(img, cax=cax, ticks=levels, format=l_f)

        mng = pylab.plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        mng.resize(700, 1000)
        if save:
            path = "hessian_approx." + extension
            if file_path is not None:
                path = file_path + "_" + path
            pylab.plt.savefig(path)
            self.output_files.append(path)
        if show:
            pylab.plt.show()
        else:
            pylab.plt.close(fig)
