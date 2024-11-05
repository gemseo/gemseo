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

from typing import ClassVar

from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import SymmetricalLogLocator
from numpy import arange
from numpy import array
from numpy import isnan
from numpy import max as np_max
from numpy import min as np_min
from numpy import ones_like

from gemseo.post.base_post import BasePost
from gemseo.post.core.colormaps import PARULA
from gemseo.post.core.hessians import SR1Approx
from gemseo.post.hessian_history_settings import HessianHistory_Settings


class HessianHistory(BasePost):
    """Plot the history of the diagonal of the Hessian matrix."""

    Settings: ClassVar[type[HessianHistory_Settings]] = HessianHistory_Settings

    def _plot(self, settings: HessianHistory_Settings) -> None:
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

        variable_names = settings.variable_names
        if variable_names:
            diag = diag[
                self.optimization_problem.design_space.get_variables_indexes(
                    variable_names
                ),
                :,
            ]

        fig, ax = plt.subplots(1, 1, figsize=settings.fig_size)
        ax.set_title("Hessian diagonal approximation")
        ax.set_xlabel("Iterations", fontsize=12)
        ax.set_yticks(arange(len(diag)))
        ax.set_yticklabels(
            self._get_design_variable_names(variable_names, simplify=True)
        )
        n_iterations = len(self.database)
        ax.set_xticks(range(n_iterations))
        ax.set_xticklabels(range(1, n_iterations + 1))
        ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))

        vmax = max(abs(np_max(diag)), abs(np_min(diag)))

        img = ax.imshow(
            diag.real,
            cmap=PARULA,
            interpolation="nearest",
            aspect="auto",
            norm=SymLogNorm(vmin=-vmax, vmax=vmax, linthresh=1.0),
        )

        color_bar = fig.colorbar(
            img,
            ticks=SymmetricalLogLocator(linthresh=1.0, base=10),
            format=LogFormatterSciNotation(),
        )
        color_bar.ax.tick_params(labelsize=9)

        plt.get_current_fig_manager().resize(700, 1000)
        fig.tight_layout()
        self._add_figure(fig, "hessian_approximation")
