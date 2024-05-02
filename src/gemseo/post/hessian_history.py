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
"""Plot the history of the diagonal of the Hessian matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import MaxNLocator
from numpy import append
from numpy import arange
from numpy import array
from numpy import concatenate
from numpy import e
from numpy import isnan
from numpy import log10 as np_log10
from numpy import logspace
from numpy import max as np_max
from numpy import min as np_min
from numpy import ones_like
from numpy import sort as np_sort

from gemseo.post.core.colormaps import PARULA
from gemseo.post.core.hessians import SR1Approx
from gemseo.post.opt_post_processor import OptPostProcessor

if TYPE_CHECKING:
    from collections.abc import Sequence


class HessianHistory(OptPostProcessor):
    """Plot the history of the diagonal of the Hessian matrix."""

    DEFAULT_FIG_SIZE = (11.0, 6.0)

    def _plot(
        self,
        variable_names: Sequence[str] = (),
    ) -> None:
        """
        Args:
            variable_names: The names of the variables to display.
                If empty, use all design variables.
        """  # noqa: D205, D212, D415
        if self.database.check_output_history_is_empty(
            self.database.get_gradient_name(self._standardized_obj_name)
        ):
            msg = (
                "The HessianHistory cannot be plotted "
                "because the history of the gradient of the objective is empty."
            )
            raise ValueError(msg)

        diag = SR1Approx(self.database).build_approximation(
            funcname=self._standardized_obj_name, save_diag=True
        )[1]
        if isnan(diag).any():
            msg = (
                "HessianHistory cannot be plotted "
                "because the approximated Hessian diagonal contains NaN."
            )
            raise ValueError(msg)

        # Add first iteration blank
        diag = array([ones_like(diag[0]), *diag]).T
        if self._change_obj:
            diag = -diag

        if variable_names:
            diag = diag[
                self.optimization_problem.design_space.get_variables_indexes(
                    variable_names
                ),
                :,
            ]

        fig = plt.figure(figsize=self.DEFAULT_FIG_SIZE)
        grid_spec = self._get_grid_layout()
        axes = fig.add_subplot(grid_spec[0, 0])
        axes.set_title("Hessian diagonal approximation")
        axes.set_xlabel("Iterations", fontsize=12)
        axes.set_yticks(arange(len(diag)))
        axes.set_yticklabels(
            self._get_design_variable_names(variable_names, simplify=True)
        )
        n_iterations = len(self.database)
        axes.set_xticks(range(n_iterations))
        axes.set_xticklabels(range(1, n_iterations + 1))
        axes.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        vmax = max(abs(np_max(diag)), abs(np_min(diag)))
        linthresh = 10 ** (np_log10(vmax) - 5.0)

        img = axes.imshow(
            diag.real,
            cmap=PARULA,
            interpolation="nearest",
            aspect="auto",
            norm=SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=linthresh, base=e),
        )

        thick_min = int(np_log10(linthresh))
        thick_max = int(np_log10(vmax))
        thick_num = thick_max - thick_min + 1
        levels_pos = append(logspace(thick_min, thick_max, num=thick_num), vmax)
        levels_neg = append(np_sort(-levels_pos), 0)
        levels = concatenate((levels_neg, levels_pos))
        color_bar = fig.colorbar(
            img,
            cax=fig.add_subplot(grid_spec[0, 1]),
            ticks=levels,
            format=LogFormatterSciNotation(),
        )
        color_bar.ax.tick_params(labelsize=9)

        plt.get_current_fig_manager().resize(700, 1000)
        self._add_figure(fig, "hessian_approximation")
