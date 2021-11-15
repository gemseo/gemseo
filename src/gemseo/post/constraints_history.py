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
"""A constraints plot."""
from __future__ import division, unicode_literals

import logging
from typing import Sequence

import numpy as np
from matplotlib import pyplot

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.core.colormaps import PARULA, RG_SEISMIC
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.utils.compatibility.matplotlib import SymLogNorm

LOGGER = logging.getLogger(__name__)


class ConstraintsHistory(OptPostProcessor):
    """Plot of the constraint function history with line charts.

    Indicate the violation with color on the background.

    The plot method requires the constraint names to plot.
    It is possible either to save the plot, to show the plot or both.
    """

    DEFAULT_FIG_SIZE = (11.0, 11.0)

    def __init__(
        self,
        opt_problem,  # type: OptimizationProblem
    ):  # type: (...) -> None
        super(ConstraintsHistory, self).__init__(opt_problem)
        self.cmap = PARULA  # "viridis"  # "jet"
        self.ineq_cstr_cmap = RG_SEISMIC  # "seismic" "PRGn_r"
        self.eq_cstr_cmap = "seismic"  # "seismic" "PRGn_r"

    def _plot(
        self,
        constraints_list,  # type: Sequence[str]
    ):  # type: (...) -> None
        """
        Args:
            constraints_list: The names of the constraints.

        Raises:
            ValueError: If a given element of `constraints_list` is not a
                function.
        """
        # retrieve the constraints values
        add_dv = False
        all_constr_names = self.opt_problem.get_constraints_names()

        for func in list(constraints_list):
            if func not in all_constr_names:
                raise ValueError(
                    "Cannot build constraints history plot, "
                    "function {} is not among the constraints names "
                    "or does not exist.".format(func)
                )

        vals, vname, _ = self.database.get_history_array(
            constraints_list, add_dv=add_dv
        )

        # harmonization of tables format because constraints can be vectorial
        # or scalars. *vals.shape[0] = iteration, *vals.shape[1] = cstr values
        vals = np.atleast_3d(vals)
        vals = vals.reshape((vals.shape[0], vals.shape[1] * vals.shape[2]))

        # prepare the main window
        nb_iter = vals.shape[0]

        n_funcs = len(vname)
        nrows = n_funcs // 2
        if 2 * nrows < n_funcs:
            nrows += 1

        fig, axes = pyplot.subplots(
            nrows=nrows,
            ncols=2,
            sharex=True,
            sharey=False,
            figsize=self.DEFAULT_FIG_SIZE,
        )

        fig.suptitle("Evolution of the constraints " + "w.r.t. iterations", fontsize=14)

        n_subplots = nrows * 2

        x_iter = np.arange(nb_iter)
        y_lim = 0.0
        vmax = 0.0
        # for each subplot
        for values, name, i in zip(vals.T, vname, np.arange(n_subplots)):
            # prepare the graph
            axe = axes.ravel()[i]
            axe.grid(True)
            axe.set_title(name)
            axe.set_xlim([0, nb_iter])
            axe.axhline(y_lim, color="k", linewidth=2)

            # plot values in lines
            axe.plot(x_iter, values)

            # Plot color bars
            cstr_matrix = np.atleast_2d(values)
            cmap = self.ineq_cstr_cmap
            vmax = max(vmax, np.max(np.abs(cstr_matrix)))
            extent = -0.5, nb_iter - 0.5, np.min(cstr_matrix), np.max(cstr_matrix)

            norm = SymLogNorm(linthresh=1.0, vmin=-vmax, vmax=vmax)

            axe.imshow(
                cstr_matrix,
                cmap=cmap,
                interpolation="nearest",
                aspect="auto",
                extent=extent,
                norm=norm,
                alpha=0.6,
            )

            # plot vertical line the last time that g(x)=0
            indices = np.where(np.diff(np.sign(values)))[0]
            if indices.size != 0:
                ind = indices[-1]
                x_lim = np.interp(y_lim, values[ind : ind + 2], x_iter[ind : ind + 2])
                axe.axvline(x_lim, color="k", linewidth=2)

        self._add_figure(fig)
