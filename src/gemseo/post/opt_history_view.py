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
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import MaxNLocator
from numpy import abs as np_abs
from numpy import append
from numpy import arange
from numpy import argmin
from numpy import array
from numpy import atleast_2d
from numpy import concatenate
from numpy import e
from numpy import full
from numpy import isnan
from numpy import log10 as np_log10
from numpy import logspace
from numpy import max as np_max
from numpy import min as np_min
from numpy import ndarray
from numpy import ones_like
from numpy import sort as np_sort
from numpy import vstack
from numpy.linalg import norm

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.post.core.colormaps import PARULA
from gemseo.post.core.colormaps import RG_SEISMIC
from gemseo.post.core.hessians import SR1Approx
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import MutableSequence
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from gemseo.algos.database import Database
    from gemseo.algos.opt_problem import OptimizationProblem

LOGGER = logging.getLogger(__name__)


class OptHistoryView(OptPostProcessor):
    """The **OptHistoryView** post processing performs separated plots.

    The design variables history, the objective function history, the history of hessian
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

    __TICK_LABEL_SIZE: Final[int] = 9
    """The font size of the tick labels."""

    __AXIS_LABEL_SIZE: Final[int] = 12
    """The font size of the axis labels."""

    def __init__(  # noqa:D107
        self,
        opt_problem: OptimizationProblem,
    ) -> None:
        super().__init__(opt_problem)
        self.cmap = PARULA
        self.ineq_cstr_cmap = RG_SEISMIC
        self.eq_cstr_cmap = "seismic"
        self.__indices = array([], dtype="int")

    def _plot(
        self,
        variable_names: Sequence[str] | None = None,
        obj_min: float | None = None,
        obj_max: float | None = None,
        obj_relative: bool = False,
    ) -> None:
        """
        Args:
            variable_names: The names of the variables to display.
                If ``None``, use all design variables.
            obj_max: The upper limit of the *y*-axis on which the objective is plotted.
                This limit must be greater than or equal
                to the maximum value of the objective history.
                If ``None``, use the maximum value of the objective history.
            obj_min: The lower limit of the *y*-axis on which the objective is plotted.
                This limit must be less than or equal
                to the minimum value of the objective history.
                If ``None``, use the minimum value of the objective history.
            obj_relative: Whether the difference
                between the objective and its initial value is plotted
                instead of the objective.
        """  # noqa: D205, D212, D415
        if variable_names:
            self.__indices = self.opt_problem.design_space.get_variables_indexes(
                variable_names
            )

        # compute the names of the inequality and equality constraints
        ineq_cstr = self.opt_problem.get_ineq_constraints()
        ineq_cstr_names = [c.name for c in ineq_cstr]
        eq_cstr = self.opt_problem.get_eq_constraints()
        eq_cstr_names = [c.name for c in eq_cstr]

        obj_history, x_history, n_iter, x_history_to_display = self._get_history(
            self._standardized_obj_name, variable_names
        )
        # design variables
        self._create_variables_plot(x_history_to_display, variable_names)

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
        if not self.database.check_output_history_is_empty(
            self.database.get_gradient_name(self._standardized_obj_name)
        ):
            self._create_hessian_approx_plot(
                self.database, self._standardized_obj_name, variable_names
            )

        # inequality and equality constraints
        self._plot_cstr_history(ineq_cstr_names, MDOFunction.ConstraintType.INEQ)
        self._plot_cstr_history(eq_cstr_names, MDOFunction.ConstraintType.EQ)

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
        variable_names: Sequence[str] | None = None,
    ) -> tuple[ndarray, ndarray, int, ndarray]:
        """Access the optimization history of a function and the design variables.

        Args:
            function_name: The name of the function.
            variable_names: The names of the variables to display.
                If ``None``, use all design variables.

        Returns:
            The history of the function outputs,
            the history of the design variables,
            the number of iterations and
            the history of the design variables to display.
        """
        f_hist, x_hist = self.database.get_function_history(
            function_name, with_x_vect=True
        )
        f_hist = array(f_hist).real
        complete_x_hist = array(x_hist).real

        x_hist_to_display = complete_x_hist
        if variable_names is not None:
            indices = [
                index
                for name in variable_names
                for index in self.opt_problem.design_space.names_to_indices[name]
            ]
            x_hist_to_display = complete_x_hist[:, indices]

        return f_hist, complete_x_hist, complete_x_hist.shape[0], x_hist_to_display

    def _get_constraints(
        self, constraint_names: MutableSequence[str]
    ) -> tuple[ndarray, list[ndarray]]:
        """Extract a history of constraints.

        Args:
            constraint_names: The names of the constraints.

        Returns:
            The bounds of the constraints and history array.
        """
        available_data_names = self.database.get_function_names()
        for constraint_name in constraint_names:
            if constraint_name not in available_data_names:
                constraint_names.remove(constraint_name)

        constraints_history = []
        bounds = full(len(constraint_names), sys.float_info.min)
        for constraint_index, constraint_name in enumerate(constraint_names):
            constraint_history = array(
                self.database.get_function_history(constraint_name)
            ).real
            constraints_history.append(constraint_history)
            bounds[constraint_index] = max(
                bounds[constraint_index],
                max(abs(np_min(constraint_history)), abs(np_max(constraint_history))),
            )

        return bounds, constraints_history

    def _normalize_x_hist(
        self, x_history: ndarray, variable_names: Sequence[str] | None
    ) -> ndarray:
        """Normalize the design variables history.

        Args:
            x_history: The history for the design variables.
            variable_names: The names of the variables to display.
                If ``None``, use all design variables.

        Returns:
            The normalized design variables array.
        """
        lower_bounds = self.opt_problem.design_space.get_lower_bounds(variable_names)
        upper_bounds = self.opt_problem.design_space.get_upper_bounds(variable_names)
        return (x_history - lower_bounds) / (upper_bounds - lower_bounds)

    def _create_variables_plot(
        self,
        x_history: ndarray,
        variable_names: Sequence[str] | None,
    ) -> None:
        """Create the design variables plot.

        Args:
            x_history: The history for the design variables.
            variable_names: The names of the variables to display.
                If ``None``, use all design variables.
        """
        n_iterations = len(x_history)
        if n_iterations < 2:
            return
        n_variables = x_history.shape[1]
        norm_x_history = self._normalize_x_hist(x_history, variable_names)

        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        grid = self._get_grid_layout()

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
        ax1.set_yticks(arange(n_variables))
        ax1.set_yticklabels(self._get_design_variable_names(variable_names, True))
        ax1.set_xlabel(self.x_label)
        # ax1.invert_yaxis()

        ax1.set_title("Evolution of the optimization variables")
        ax1.set_xticks(list(range(n_iterations)))
        ax1.set_xticklabels(list(range(1, n_iterations + 1)))
        ax1.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        # colorbar
        ax2 = fig.add_subplot(grid[0, 1])
        fig.colorbar(im1, cax=ax2)

        # Set window size
        mng = plt.get_current_fig_manager()
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
                If ``None``, use the maximum value of the objective history.
            obj_min: The minimum value for the objective in the plot.
                If ``None``, use the minimum value of the objective history.
            obj_relative: If ``True``, plot the objective value difference
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
        n_iterations = len(obj_history)
        x_absc = arange(n_iterations)
        x_absc_nan = None
        idx_nan = isnan(obj_history)

        if idx_nan.size > 0:
            obj_history = obj_history[~idx_nan]
            x_absc_nan = x_absc[idx_nan]
            x_absc_not_nan = x_absc[~idx_nan]

        fmin = np_min(obj_history)
        fmax = np_max(obj_history)

        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        # objective function
        plt.xlabel(self.x_label, fontsize=self.__AXIS_LABEL_SIZE)
        plt.ylabel("Objective value", fontsize=self.__AXIS_LABEL_SIZE)

        plt.plot(x_absc_not_nan, obj_history)

        if idx_nan.size > 0:
            for x_i in x_absc_nan:
                plt.axvline(x_i, color="purple")

        if obj_min is not None and obj_min < fmin:
            fmin = obj_min
        if obj_max is not None and obj_max > fmax:
            fmax = obj_max

        margin = (fmax - fmin) * self._Y_MARGIN
        plt.ylim([fmin - margin, fmax + margin])
        plt.xlim([0 - self._X_MARGIN, n_iter - 1 + self._X_MARGIN])
        ax1 = fig.gca()
        ax1.set_xticks(x_absc)
        ax1.set_xticklabels((x_absc + 1).tolist())
        ax1.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.title("Evolution of the objective value")

        # Set window size
        mng = plt.get_current_fig_manager()
        mng.resize(700, 1000)

        self._add_figure(fig, "objective")

    def _create_x_star_plot(
        self,
        x_history: ndarray,
        n_iter: int,
    ) -> None:
        """Create the design variables plot.

        Args:
            x_history: The history of the design variables.
            n_iter: The number of iterations.
        """
        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        plt.xlabel(self.x_label, fontsize=self.__AXIS_LABEL_SIZE)
        plt.ylabel("||x-x*||", fontsize=self.__AXIS_LABEL_SIZE)
        normalize = self.opt_problem.design_space.normalize_vect
        x_xstar = norm(
            normalize(x_history) - normalize(self.opt_problem.get_optimum()[1]), axis=1
        )

        # Draw a vertical line at the optimum
        n_iterations = len(x_history)
        plt.axvline(x=argmin(x_xstar), color="r")
        plt.semilogy(arange(n_iterations), x_xstar)
        # ======================================================================
        # try:
        #     plt.semilogy(np.arange(len(x_xstar)), x_xstar)
        # except ValueError:
        #     LOGGER.warning("Cannot use log scale for x_star plot since" +
        #                    "all values are not positive !")
        # ======================================================================
        ax1 = fig.gca()
        ax1.set_xticks(range(n_iterations))
        ax1.set_xticklabels(range(1, n_iterations + 1))
        ax1.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.title("Distance to the optimum")
        plt.xlim([0 - self._X_MARGIN, n_iter - 1 + self._X_MARGIN])

        # Set window size
        mng = plt.get_current_fig_manager()
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

        for i, cstr_history_i in enumerate(cstr_history):
            history_i = atleast_2d(cstr_history_i).T
            if history_i.shape[1] == 1:
                history_i = history_i.T

            nb_components = history_i.shape[0]

            if history_i.shape[1] == max_iter:  # TEST
                for component_j in range(nb_components):
                    # compute the label of the constraint
                    if component_j == 0:
                        cstr_label = repr_variable(
                            cstr_names[i], component_j, nb_components
                        )
                    else:
                        cstr_label = repr_variable("", component_j)

                    cstr_labels.append(cstr_label)
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
        cstr_type: MDOFunction.ConstraintType,
        vmax: float,
        n_cstr: int,
        cstr_labels: Sequence[str],
    ) -> Figure:
        """Build the constraints figure.

        Args:
            cstr_matrix: The matrix of constraints values.
            cstr_type: The type of the constraints.
            cstr_labels: The labels for the constraints.
            vmax: The maximum constraint absolute value.
            n_cstr: The number of constraints.
            cstr_labels: The labels of constraints names.

        Returns:
            The constraints figure.
        """
        if cstr_type == MDOFunction.ConstraintType.EQ:
            cmap = self.eq_cstr_cmap
            constraint_type = "equality"
        else:
            cmap = self.ineq_cstr_cmap
            constraint_type = "inequality"

        idx_nan = isnan(cstr_matrix)
        hasnan = idx_nan.any()
        if hasnan > 0:
            cstr_matrix[idx_nan] = 0.0

        # generation of the image
        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        grid = self._get_grid_layout()
        ax1 = fig.add_subplot(grid[0, 0])
        im1 = ax1.imshow(
            cstr_matrix,
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
            norm=SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=1.0, base=e),
        )
        if hasnan > 0:
            x_absc_nan = idx_nan.any(axis=0).nonzero()[0]
            for x_i in x_absc_nan:
                plt.axvline(x_i, color="purple")

        ax1.tick_params(labelsize=self.__TICK_LABEL_SIZE)
        ax1.set_yticks(list(range(n_cstr)))
        ax1.set_yticklabels(cstr_labels)

        ax1.set_xlabel(self.x_label)
        ax1.set_title(f"Evolution of the {constraint_type} constraints")
        n_iterations = len(self.database)
        ax1.set_xticks(range(n_iterations))
        ax1.set_xticklabels(range(1, n_iterations + 1))

        ax1.hlines(
            list(range(len(cstr_matrix))),
            [-0.5],
            [len(cstr_matrix[0]) - 0.5],
            alpha=0.1,
            lw=0.5,
        )
        ax1.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        # color map
        cax = fig.add_subplot(grid[0, 1])
        thick_min = int(np_log10(vmax)) if 0.0 < vmax < 1.0 else 0
        thick_max = int(np_log10(vmax)) if vmax > 1.0 else 0
        thick_num = thick_max - thick_min + 1
        levels_pos = logspace(thick_min, thick_max, num=thick_num)
        if vmax != 0.0:
            levels_pos = np_sort(append(levels_pos, vmax))
        levels_neg = np_sort(-levels_pos)
        levels_neg = append(levels_neg, 0)
        levels = concatenate((levels_neg, levels_pos))
        col_bar = fig.colorbar(
            im1, cax=cax, ticks=levels, format=LogFormatterSciNotation()
        )
        col_bar.ax.tick_params(labelsize=self.__TICK_LABEL_SIZE)

        mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        mng.resize(700, 1000)
        return fig

    def _create_hessian_approx_plot(
        self,
        history: Database,
        obj_name: str,
        variable_names: Sequence[str] | None,
    ) -> None:
        """Create the plot of the Hessian approximation.

        Args:
            history: The optimization history.
            obj_name: The objective function name.
            variable_names: The names of the variables to display.
                If ``None``, use all design variables.
        """
        try:
            diag = SR1Approx(history).build_approximation(
                funcname=obj_name, save_diag=True
            )[1]
        except ValueError:
            LOGGER.warning("Failed to create Hessian approximation.", exc_info=True)
            return

        if isnan(diag).any():
            LOGGER.warning("Failed to create Hessian approximation.")
            LOGGER.warning("The approximated Hessian diagonal contains NaN.")
            return

        diag = [ones_like(diag[0]), *diag]  # Add first iteration blank
        diag = array(diag).T

        # if max problem, plot -Hessian
        if self._change_obj:
            diag = -diag

        if variable_names:
            diag = diag[self.__indices, :]

        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        grid = self._get_grid_layout()

        axe = fig.add_subplot(grid[0, 0])
        axe.set_title("Hessian diagonal approximation")
        axe.set_xlabel(self.x_label, fontsize=self.__AXIS_LABEL_SIZE)
        axe.set_yticks(arange(len(diag)))
        axe.set_yticklabels(
            self._get_design_variable_names(variable_names, simplify=True)
        )
        n_iterations = len(self.database)
        axe.set_xticks(range(n_iterations))
        axe.set_xticklabels(range(1, n_iterations + 1))
        axe.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        # matrix
        vmax = max(abs(np_max(diag)), abs(np_min(diag)))
        linthresh = 10 ** (np_log10(vmax) - 5.0)
        img = axe.imshow(
            diag.real,
            cmap=self.cmap,
            interpolation="nearest",
            aspect="auto",
            norm=SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, base=e),
        )

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

        cax = fig.add_subplot(grid[0, 1])
        col_bar = fig.colorbar(
            img, cax=cax, ticks=levels, format=LogFormatterSciNotation()
        )
        col_bar.ax.tick_params(labelsize=self.__TICK_LABEL_SIZE)

        mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        mng.resize(700, 1000)
        self._add_figure(fig, "hessian_approximation")
