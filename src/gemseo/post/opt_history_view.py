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
"""Basic display of optimization history: functions and x."""
from __future__ import annotations

import logging
import sys
from typing import ClassVar
from typing import Iterable
from typing import MutableSequence
from typing import Sequence

import matplotlib.gridspec as gridspec
import pylab
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatter
from matplotlib.ticker import MaxNLocator
from numpy import abs as np_abs
from numpy import append
from numpy import arange
from numpy import argmin
from numpy import array
from numpy import atleast_2d
from numpy import concatenate
from numpy import full
from numpy import hstack
from numpy import isnan
from numpy import log10 as np_log10
from numpy import logspace
from numpy import max as np_max
from numpy import min as np_min
from numpy import ndarray
from numpy import ones_like
from numpy import sort as np_sort
from numpy import vstack
from numpy import where
from numpy.linalg import norm

from gemseo.algos.database import Database
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.post.core.colormaps import PARULA
from gemseo.post.core.colormaps import RG_SEISMIC
from gemseo.post.core.hessians import SR1Approx
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.compatibility.matplotlib import SymLogNorm

LOGGER = logging.getLogger(__name__)


class OptHistoryView(OptPostProcessor):
    """The **OptHistoryView** post processing performs separated plots.

    The design
    variables history, the objective function history, the history of hessian
    approximation of the objective, the inequality constraint history, the equality
    constraint history, and constraints histories.

    By default, all design variables are considered. A sublist of design variables can
    be passed as options. Minimum and maximum values for the plot can be passed as
    options. The objective function can also be represented in terms of difference
    w.r.t. the initial value. It is possible either to save the plot, to show the plot
    or both.
    """

    DEFAULT_FIG_SIZE = (11.0, 6.0)

    x_label: ClassVar[str] = "Iterations"
    """The label for the x-axis."""

    def __init__(  # noqa:D107
        self,
        opt_problem: OptimizationProblem,
    ) -> None:
        super().__init__(opt_problem)
        self.cmap = PARULA
        self.ineq_cstr_cmap = RG_SEISMIC
        self.eq_cstr_cmap = "seismic"

    def _plot(
        self,
        variables_names: Sequence[str] | None = None,
        obj_min: float | None = None,
        obj_max: float | None = None,
        obj_relative: bool = False,
    ) -> None:
        """
        Args:
            variables_names: The names of the variables to display.
                If None, use all design variables.
            obj_max: The maximum value for the objective in the plot.
                If None, use the maximum value of the objective history.
            obj_min: The minimum value for the objective in the plot.
                If None, use the minimum value of the objective history.
            obj_relative: If True, plot the objective value difference
                with the initial value.
        """  # noqa: D205, D212, D415
        # compute the names of the inequality and equality constraints
        ineq_cstr = self.opt_problem.get_ineq_constraints()
        ineq_cstr_names = [c.name for c in ineq_cstr]
        eq_cstr = self.opt_problem.get_eq_constraints()
        eq_cstr_names = [c.name for c in eq_cstr]

        obj_history, x_history, n_iter = self._get_history(
            self._standardized_obj_name, variables_names
        )

        # design variables
        self._create_variables_plot(x_history, variables_names)

        # objective function
        self._create_obj_plot(
            obj_history,
            n_iter,
            obj_min,
            obj_max,
            obj_relative=obj_relative,
        )

        self._create_x_star_plot(x_history, n_iter)

        # Hessian plot
        plot_hessian = not self.database.is_func_grad_history_empty(
            self._standardized_obj_name
        )
        if plot_hessian:
            self._create_hessian_approx_plot(self.database, self._standardized_obj_name)

        # inequality and equality constraints
        self._plot_cstr_history(ineq_cstr_names, MDOFunction.TYPE_INEQ)
        self._plot_cstr_history(eq_cstr_names, MDOFunction.TYPE_EQ)

    def _plot_cstr_history(
        self,
        cstr_names: MutableSequence[str],
        cstr_type: str,
    ) -> None:
        """Create the plot for (in)equality constraints.

        Args:
            cstr_names: The names of the constraints.
            cstr_type: The type of the constraints, either 'eq' or 'ineq'.
        """
        if cstr_names is not None:
            _, constraints_history = self._get_constraints(cstr_names)
            self._create_cstr_plot(
                constraints_history,
                cstr_type,
                cstr_names,
            )

    def _get_history(
        self,
        function_name: str,
        variables_names: Sequence[str],
    ) -> tuple[ndarray, ndarray, int]:
        """Access the optimization history of a function and the design variables.

        This is done at which it was computed.

        Args:
            function_name: The name of the function.
            variables_names: The names of the variables to display.

        Returns:
            The function values,
            the design variables values
            and the number of iterations.
        """
        f_hist, x_hist = self.database.get_func_history(function_name, x_hist=True)
        f_hist = array(f_hist).real
        x_hist = array(x_hist).real

        if variables_names is not None:
            # select only the interesting columns
            blocks = []
            column = 0
            for var in self.opt_problem.design_space.variables_names:
                if var in variables_names:
                    size = self.opt_problem.design_space.variables_sizes[var]
                    blocks.append(x_hist[:, column : column + size])
                    column += size
            # concatenate the blocks
            x_hist = hstack(blocks)

        return f_hist, x_hist, x_hist.shape[0]

    def _get_constraints(
        self, constraint_names: MutableSequence[str]
    ) -> tuple[ndarray, list[ndarray]]:
        """Extract a history of constraints.

        Args:
            constraint_names: The names of the constraints.

        Returns:
            The bounds of the constraints and history array.
        """
        available_data_names = self.database.get_all_data_names()
        for constraint_name in constraint_names:
            if constraint_name not in available_data_names:
                constraint_names.remove(constraint_name)

        constraints_history = []
        bounds = full(len(constraint_names), sys.float_info.min)
        for constraint_index, constraint_name in enumerate(constraint_names):
            constraint_history = array(
                self.database.get_func_history(constraint_name)
            ).real
            constraints_history.append(constraint_history)
            bounds[constraint_index] = max(
                bounds[constraint_index],
                max(abs(np_min(constraint_history)), abs(np_max(constraint_history))),
            )

        return bounds, constraints_history

    def _normalize_x_hist(
        self,
        x_history: ndarray,
        variables_names: Sequence[str],
    ) -> ndarray:
        """Normalize the design variables history.

        Args:
            x_history: The history for the design variables.
            variables_names: The names of the variables to display.

        Returns:
            The normalized design variables array.
        """
        x_hist_n = x_history.copy()
        lower_bounds = self.opt_problem.design_space.get_lower_bounds(variables_names)
        upper_bounds = self.opt_problem.design_space.get_upper_bounds(variables_names)
        norm_coeff = 1 / (np_abs(upper_bounds - lower_bounds))
        for i in range(x_history.shape[0]):
            x_hist_n[i, :] = (x_hist_n[i, :] - lower_bounds) * norm_coeff
        return x_hist_n

    def _create_variables_plot(
        self,
        x_history: ndarray,
        variables_names: Sequence[str],
    ) -> None:
        """Create the design variables plot.

        Args:
             x_history: The history for the design variables.
             variables_names: The names of the variables to display.
        """
        if len(x_history) < 2:
            return
        n_variables = x_history.shape[1]
        norm_x_history = self._normalize_x_hist(x_history, variables_names)

        fig = pylab.plt.figure(figsize=self.DEFAULT_FIG_SIZE)
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
                y_labels.append(f"({i})")

        ax1.set_yticks(arange(n_variables))
        ax1.set_yticklabels(y_labels)
        ax1.set_xlabel(self.x_label)
        # ax1.invert_yaxis()

        ax1.set_title("Evolution of the optimization variables")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # colorbar
        ax2 = fig.add_subplot(grid[0, 1])
        fig.colorbar(im1, cax=ax2)

        # Set window size
        mng = pylab.plt.get_current_fig_manager()
        mng.resize(700, 1000)

        self._add_figure(fig, "variables")

    def _create_obj_plot(
        self,
        obj_history: ndarray,
        n_iter: int,
        obj_min: float | None = None,
        obj_max: float | None = None,
        obj_relative: bool = False,
    ) -> None:
        """Creates the design variables plot.

        Args:
            obj_history: The history of the objective function.
            n_iter: The number of iterations.
            obj_max: The maximum value for the objective in the plot.
                If None, use the maximum value of the objective history.
            obj_min: The minimum value for the objective in the plot.
                If None, use the minimum value of the objective history.
            obj_relative: If True, plot the objective value difference
                with the initial value.
        """
        if self._change_obj:
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

        fig = pylab.plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        # objective function
        pylab.plt.xlabel(self.x_label, fontsize=12)
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

        # Set window size
        mng = pylab.plt.get_current_fig_manager()
        mng.resize(700, 1000)

        self._add_figure(fig, "objective")

    def _create_x_star_plot(
        self,
        x_history: ndarray,
        n_iter: int,
    ):
        """Create the design variables plot.

        Args:
            x_history: The history of the design variables.
            n_iter: The number of iterations.
        """
        fig = pylab.plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        # objective function
        pylab.plt.xlabel(self.x_label, fontsize=12)
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

        # Set window size
        mng = pylab.plt.get_current_fig_manager()
        mng.resize(700, 1000)

        self._add_figure(fig, "x_xstar")

    @staticmethod
    def _cstr_number(constraint_history: Iterable[ndarray]) -> int:
        """Compute the total scalar constraints number.

        Args:
            constraint_history: The history of the constraints.

        Returns:
            The number of constraints.
        """
        n_cstr = 0
        for constraint_history_i in constraint_history:
            c_hist_loc = atleast_2d(constraint_history_i).T
            if c_hist_loc.shape[1] == 1:
                c_hist_loc = c_hist_loc.T
            n_cstr += c_hist_loc.shape[0]
        LOGGER.debug("Total constraints number =%s", n_cstr)
        return n_cstr

    def _create_cstr_plot(
        self,
        cstr_history: Iterable[ndarray],
        cstr_type: str,
        cstr_names: Sequence[str],
    ) -> None:
        """Create the constraints plot: 1 line per constraint component.

        Args:
            cstr_history: The history of the constraints.
            cstr_type: The type of the constraints, either 'eq' or 'ineq'.
            cstr_names: The names of the constraints.
        """
        n_cstr = self._cstr_number(cstr_history)
        if n_cstr == 0:
            return

        # matrix of all constraints' values
        cstr_matrix = None
        vmax = 0.0
        cstr_labels = []

        max_iter = 0
        for cstr_history_i in cstr_history:
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
                            cstr_name += " (" + str(component_j) + ")"
                        cstr_labels.append(cstr_name)
                    else:
                        cstr_labels.append("(" + str(component_j) + ")")

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

        self._add_figure(fig, f"{cstr_type}_constraints")

    def _build_cstr_fig(
        self,
        cstr_matrix: ndarray,
        cstr_type: str,
        vmax: float,
        n_cstr: int,
        cstr_labels: Sequence[str],
    ) -> Figure:
        """Build the constraints figure.

        Args:
            cstr_matrix: The matrix of constraints values.
            cstr_type: The type of the constraints, either 'eq' or 'ineq'.
            cstr_labels: The labels for the constraints.
            vmax: The maximum constraint absolute value.
            n_cstr: The number of constraints.
            cstr_labels: The labels of constraints names.

        Returns:
            The constraints figure.
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
        fig = pylab.plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        grid = gridspec.GridSpec(1, 2, width_ratios=[15, 1], wspace=0.04, hspace=0.6)
        ax1 = fig.add_subplot(grid[0, 0])
        im1 = ax1.imshow(
            cstr_matrix,
            cmap=cmap,
            interpolation="nearest",
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

        ax1.set_xlabel(self.x_label)
        ax1.set_title(f"Evolution of the {fullname} constraints")
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
        levels_pos = logspace(thick_min, thick_max, num=thick_num)
        if vmax != 0.0:
            levels_pos = np_sort(append(levels_pos, vmax))
        levels_neg = np_sort(-levels_pos)
        levels_neg = append(levels_neg, 0)
        levels = concatenate((levels_neg, levels_pos))
        col_bar = fig.colorbar(im1, cax=cax, ticks=levels)
        col_bar.ax.tick_params(labelsize=9)

        mng = pylab.plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        mng.resize(700, 1000)
        return fig

    def _create_hessian_approx_plot(
        self,
        history: Database,
        obj_name: str,
    ) -> None:
        """Create the plot of the Hessian approximation.

        Args:
            history: The optimization history.
            obj_name: The objective function name.
        """
        try:
            approximator = SR1Approx(history)
            _, diag, _, _ = approximator.build_approximation(
                funcname=obj_name, save_diag=True
            )
            if isnan(diag).any():
                raise ValueError("The approximated Hessian diagonal contains NaN.")

            diag = [ones_like(diag[0])] + diag  # Add first iteration blank
            diag = array(diag).T
        except ValueError:
            LOGGER.warning("Failed to create Hessian approximation.", exc_info=True)
            return

        # if max problem, plot -Hessian
        if not self.opt_problem.minimize_objective:
            diag = -diag

        fig = pylab.plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        grid = gridspec.GridSpec(1, 2, width_ratios=[15, 1], wspace=0.04, hspace=0.6)
        # matrix
        axe = fig.add_subplot(grid[0, 0])

        axe.set_title("Hessian diagonal approximation")
        axe.set_xlabel(self.x_label, fontsize=12)
        axe.set_ylabel("Variable id", fontsize=12)
        axe.xaxis.set_major_locator(MaxNLocator(integer=True))
        vmax = max(abs(np_max(diag)), abs(np_min(diag)))
        linthresh = 10 ** (np_log10(vmax) - 5.0)
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
        levels_pos = logspace(thick_min, thick_max, num=thick_num)
        levels_pos = append(levels_pos, vmax)
        levels_neg = np_sort(-levels_pos)
        levels_neg = append(levels_neg, 0)
        levels = concatenate((levels_neg, levels_pos))

        l_f = LogFormatter(base=10)
        cax = fig.add_subplot(grid[0, 1])
        cax.invert_yaxis()
        fig.colorbar(img, cax=cax, ticks=levels, format=l_f)

        mng = pylab.plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        mng.resize(700, 1000)
        self._add_figure(fig, "hessian_approximation")
