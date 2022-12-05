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
"""A constraints plot."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from matplotlib import pyplot

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.core.colormaps import PARULA
from gemseo.post.core.colormaps import RG_SEISMIC
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.compatibility.matplotlib import SymLogNorm


class ConstraintsHistory(OptPostProcessor):
    """Plot of the constraint function history with line charts.

    Indicate the violation with color on the background.

    The plot method requires the constraint names to plot.
    It is possible either to save the plot, to show the plot or both.
    """

    DEFAULT_FIG_SIZE = (11.0, 11.0)

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
        constraint_names: Sequence[str],
    ) -> None:
        """
        Args:
            constraint_names: The names of the constraints.

        Raises:
            ValueError: If a given element of `constraint_names` is not a function.
        """  # noqa: D205, D212, D415
        all_constraint_names = self.opt_problem.constraint_names.keys()
        for constraint_name in constraint_names:
            if constraint_name not in all_constraint_names:
                raise ValueError(
                    "Cannot build constraints history plot, "
                    f"function {constraint_name} is not among the constraints names "
                    "or does not exist."
                )

        constraint_names = self.opt_problem.get_function_names(constraint_names)
        constraint_history, constraint_names, _ = self.database.get_history_array(
            constraint_names, add_dv=False
        )

        # harmonization of tables format because constraints can be vectorial
        # or scalars. *vals.shape[0] = iteration, *vals.shape[1] = cstr values
        constraint_history = np.atleast_3d(constraint_history)
        constraint_history = constraint_history.reshape(
            (
                constraint_history.shape[0],
                constraint_history.shape[1] * constraint_history.shape[2],
            )
        )

        # prepare the main window
        n_iterations = len(constraint_history)

        n_funcs = len(constraint_names)
        n_rows = n_funcs // 2
        if 2 * n_rows < n_funcs:
            n_rows += 1

        fig, axes = pyplot.subplots(
            nrows=n_rows, ncols=2, sharex=True, figsize=self.DEFAULT_FIG_SIZE
        )

        fig.suptitle("Evolution of the constraints w.r.t. iterations", fontsize=14)

        n_subplots = n_rows * 2

        iterations = np.arange(n_iterations)
        # for each subplot
        for history, name, i in zip(
            constraint_history.T, constraint_names, np.arange(n_subplots)
        ):
            # prepare the graph
            axe = axes.ravel()[i]
            axe.grid(True)
            axe.set_title(name)
            axe.set_xlim([0, n_iterations])
            axe.axhline(0.0, color="k", linewidth=2)

            # plot values in lines
            axe.plot(iterations, history)

            # Plot color bars
            maximum = np.max(np.abs(history))
            axe.imshow(
                np.atleast_2d(history),
                cmap=self.ineq_cstr_cmap,
                interpolation="nearest",
                aspect="auto",
                extent=[-0.5, n_iterations - 0.5, np.min(history), np.max(history)],
                norm=SymLogNorm(linthresh=1.0, vmin=-maximum, vmax=maximum),
                alpha=0.6,
            )

            # plot vertical line the last time that g(x)=0
            indices = np.where(np.diff(np.sign(history)))[0]
            if indices.size != 0:
                ind = indices[-1]
                x_lim = np.interp(
                    0.0, history[ind : ind + 2], iterations[ind : ind + 2]
                )
                axe.axvline(x_lim, color="k", linewidth=2)

        self._add_figure(fig)
