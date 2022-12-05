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
"""History of the maximum constraint and objective value."""
from __future__ import annotations

import logging
from typing import Sequence

import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import ndarray

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.post.core.colormaps import PARULA
from gemseo.post.core.colormaps import RG_SEISMIC
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.compatibility.matplotlib import SymLogNorm

LOGGER = logging.getLogger(__name__)


class ObjConstrHist(OptPostProcessor):
    """History of the maximum constraint and objective value.

    The objective history is plotted with a line
    over the maximum constraint history plotted with the green-white-red color bar:

    - white: the constraint is active;
    - green: the equality constraint is violated while the inequality one is satisfied;
    - red: the inequality constraint is violated.
    """

    DEFAULT_FIG_SIZE = (11.0, 6.0)

    def __init__(  # noqa:D107
        self,
        opt_problem: OptimizationProblem,
    ) -> None:
        super().__init__(opt_problem)
        self.opt_problem = opt_problem
        self.cmap = PARULA
        self.ineq_cstr_cmap = RG_SEISMIC
        self.eq_cstr_cmap = "seismic"

    def _plot(
        self,
        constraint_names: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            constraint_names: The names of the constraints to plot.
                If ``None``, use all the constraints.
        """  # noqa: D205, D212, D415
        # 0. Initialize the figure.
        grid = gridspec.GridSpec(1, 2, width_ratios=[15, 1], wspace=0.04, hspace=0.6)
        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        ax1 = fig.add_subplot(grid[0, 0])
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        mng = plt.get_current_fig_manager()
        mng.resize(700, 1000)

        # 1. Plot the objective history versus the iterations with a curve.
        problem = self.opt_problem
        obj_history, x_history = self.database.get_func_history(
            problem.get_objective_name(), x_hist=True
        )
        obj_history, x_history = np.array(obj_history).real, np.array(x_history).real
        obj_min, obj_max = obj_history.min(), obj_history.max()
        if not problem.minimize_objective and problem.use_standardized_objective:
            obj_history = -obj_history
            obj_min, obj_max = obj_history.min(), obj_history.max()

        plt.plot(obj_history)
        plt.xlabel("Iterations", fontsize=12)
        plt.ylabel("Objective value", fontsize=12)
        plt.ylim([obj_min, obj_max])
        plt.xlim([0, len(x_history)])
        plt.grid(True)
        plt.title("Evolution of the objective value and maximal constraint")

        # 2. Plot the maximum constraint history versus the iterations
        #    with green-white-red color map.
        # 2.a. Get inequality and equality constraint histories.
        ineq_history, ineq_names = self.__get_constraints(
            problem.get_ineq_constraints(), constraint_names
        )
        eq_history, eq_names = self.__get_constraints(
            problem.get_eq_constraints(), constraint_names
        )
        # 2.b. Concatenate the inequality and equality constraint histories.
        #      NB: Take absolute values of equality constraints for color map.
        constraint_history = np.concatenate(
            [
                constraint_history
                for constraint_history in [ineq_history, np.abs(eq_history)]
                if constraint_history.size > 0
            ],
            axis=1,
        )
        c_max = abs(constraint_history).max()
        im1 = ax1.imshow(
            np.atleast_2d(np.amax(constraint_history, axis=1)),
            cmap=RG_SEISMIC,
            interpolation="nearest",
            aspect="auto",
            extent=[-0.5, len(x_history) - 0.5, obj_min, obj_max],
            norm=SymLogNorm(linthresh=1.0, vmin=-c_max, vmax=c_max),
        )
        # 2.c. Add vertical labels with constraint violation information.
        constraint_names = np.concatenate((ineq_names, eq_names))
        constraint_values = np.concatenate(
            [values for values in [ineq_history, eq_history] if values.size > 0], axis=1
        )
        ordinate = obj_min + (obj_max + obj_min) / 2 * 0.1
        for iteration, i in enumerate(np.argmax(constraint_history, axis=1)):
            ax1.text(
                iteration + 0.05,
                ordinate,
                f"constraint {constraint_names[i]} = {constraint_values[iteration, i]:.2e}",
                rotation="vertical",
            )
        # 2.d. Add color map.
        thick_max = int(np.log10(np.abs(c_max)))
        levels_pos = np.append(
            np.logspace(0, thick_max, num=thick_max + 1),
            c_max,
        )
        cax = fig.add_subplot(grid[0, 1])
        col_bar = fig.colorbar(
            im1,
            cax=cax,
            ticks=np.concatenate((np.append(np.sort(-levels_pos), 0), levels_pos)),
            format="%.2e",
        )
        col_bar.ax.tick_params(labelsize=9)
        self._add_figure(fig)

    def __get_constraints(
        self, constraints: list[MDOFunction], all_constraint_names: Sequence[str] | None
    ) -> tuple[ndarray, ndarray]:
        """Return the constraints with formatted shape.

        Args:
            constraints: The different constraints.
            all_constraint_names: The names of the constraints.
                If ``None``, use all the constraints.

        Returns:
            The history and the names of constraints.
        """
        constraint_names = []
        for constraint in constraints:
            if all_constraint_names is None or constraint.name in all_constraint_names:
                constraint_names.append(constraint.name)

        if constraint_names:
            constraint_history, constraint_names, _ = self.database.get_history_array(
                constraint_names, add_dv=False
            )
        else:
            constraint_history, constraint_names = np.array([]), np.array([])

        # harmonization of tables format because constraints can be vectorial
        # or scalars. *vals.shape[0] = iteration, *vals.shape[1] = cstr values
        constraint_history = np.atleast_3d(constraint_history)
        shape = constraint_history.shape
        constraint_history = np.reshape(
            constraint_history, (shape[0], shape[1] * shape[2])
        )
        return constraint_history, constraint_names
