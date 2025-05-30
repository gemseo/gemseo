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

from __future__ import annotations

import logging
import re
from functools import partial
from re import fullmatch
from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from numpy import atleast_2d
from numpy.core.shape_base import hstack

from gemseo.post.base_post import BasePost
from gemseo.post.correlations_settings import Correlations_Settings

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

    from gemseo.typing import NumberArray
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


class Correlations(BasePost[Correlations_Settings]):
    """Scatter plots of the correlated variables.

    These variables can be design variables, constraints, objective or observables. This
    post-processor considers all the correlations greater than a threshold.
    """

    MAXIMUM_CORRELATION_COEFFICIENT: ClassVar[float] = 1.0 - 1e-9
    """The maximum correlation coefficient above which the variable is not plotted."""

    Settings: ClassVar[type[Correlations_Settings]] = Correlations_Settings

    def _plot(self, settings: Correlations_Settings) -> None:
        """
        Raises:
            ValueError: If an element of ``func_names`` is unknown.
        """  # noqa: D205, D212, D415
        func_names = settings.func_names
        coeff_limit = settings.coeff_limit
        n_plots_x = settings.n_plots_x
        n_plots_y = settings.n_plots_y
        optimization_metadata = self._optimization_metadata

        all_func_names = (
            self._dataset.equality_constraint_names
            + self._dataset.inequality_constraint_names
            + self._dataset.objective_names
            + self._dataset.observable_names
        )
        if not func_names:
            func_names = all_func_names

        dict_ = optimization_metadata.output_names_to_constraint_names
        func_names = [
            names
            for func_name in func_names
            for names in dict_.get(func_name, [func_name])
        ]
        if (
            not optimization_metadata.minimize_objective
            and optimization_metadata.objective_name in func_names
        ):
            func_names[func_names.index(optimization_metadata.objective_name)] = (
                optimization_metadata.standardized_objective_name
            )

        not_func_names = set(func_names) - set(all_func_names)
        if not_func_names:
            msg = (
                f"The following elements are not functions: {sorted(not_func_names)}; "
                f"available ones are: {sorted(all_func_names)}."
            )
            raise ValueError(msg)

        dataset = self._dataset
        variable_history = dataset.get_view(variable_names=func_names).to_numpy()
        variable_history[np.isnan(variable_history)] = 0.0
        x_history = dataset.design_dataset.to_numpy()
        variable_history = hstack((variable_history, x_history))
        x_names = dataset.get_columns(
            dataset.get_variable_names(group_name=dataset.DESIGN_GROUP)
        )
        variable_names = dataset.get_columns(func_names)
        variable_names.extend(x_names)
        variable_names = self.__sort_variable_names(variable_names, func_names)

        if (
            optimization_metadata.standardized_objective_name in variable_names
            and self._change_obj
        ):
            obj_index = variable_names.index(
                optimization_metadata.standardized_objective_name
            )
            variable_history[:, obj_index] = -variable_history[:, obj_index]
            variable_names[obj_index] = optimization_metadata.objective_name

        correlation_coefficients = self.__compute_correlations(variable_history)
        i_corr, j_corr = (
            (np.abs(correlation_coefficients) > coeff_limit)
            & (np.abs(correlation_coefficients) < self.MAXIMUM_CORRELATION_COEFFICIENT)
        ).nonzero()
        LOGGER.info("Detected %s correlations > %s", i_corr.size, coeff_limit)

        if i_corr.size <= 16:
            n_plots_x = n_plots_y = 4

        spec = GridSpec(n_plots_y, n_plots_x, wspace=0.3, hspace=0.75)
        spec.update(top=0.95, bottom=0.06, left=0.08, right=0.95)

        for plot_index, (i, j) in enumerate(zip(i_corr, j_corr)):
            plot_index_loc = plot_index % (n_plots_x * n_plots_y)
            if plot_index_loc == 0:
                fig: Figure
                fig = plt.figure(figsize=settings.fig_size)
                mng = plt.get_current_fig_manager()
                if mng is not None:
                    mng.resize(1200, 900)
                ticker.MaxNLocator(nbins=3)

                fig.tight_layout()
                self._add_figure(fig)

            self.__create_sub_correlation_plot(
                i,
                j,
                correlation_coefficients[i, j],
                fig,
                spec,
                plot_index_loc,
                n_plots_y,
                n_plots_x,
                variable_history,
                variable_names,
            )

    def __create_sub_correlation_plot(
        self,
        x_index: int,
        y_index: int,
        correlation_coefficients: RealArray,
        fig: Figure,
        spec: GridSpec,
        plot_index: int,
        n_y: int,
        n_x: int,
        variable_history: NumberArray,
        variable_names: Sequence[str],
    ) -> None:
        """Create a correlation plot.

        Args:
            x_index: The position of the variable on the x-axis.
            y_index: The position of the variable on the y-axis.
            correlation_coefficients: The correlation coefficients.
            fig: The figure where the subplot will be placed.
            spec: The matplotlib grid structure.
            plot_index: The local plot index.
            n_y: The number of vertical plots.
            n_x: The number of horizontal plots.
            variable_history: The history of the variables.
            variable_names: The names of the variables.
        """
        ax1 = fig.add_subplot(spec[int(plot_index / n_y), plot_index % n_x])
        ax1.scatter(
            variable_history[:, x_index], variable_history[:, y_index], c="b", s=30
        )
        self.materials_for_plotting[x_index, y_index] = (
            variable_names[x_index],
            variable_names[y_index],
            correlation_coefficients,
        )
        ax1.set_xlabel(variable_names[x_index], fontsize=9)
        # Update y labels spacing
        start, stop = ax1.get_ylim()
        ax1.yaxis.set_ticks(np.arange(start, stop, 0.24999999 * (stop - start)))
        start, stop = ax1.get_xlim()
        ax1.xaxis.set_ticks(np.arange(start, stop, 0.24999999 * (stop - start)))
        ax1.set_ylabel(variable_names[y_index], fontsize=10)
        ax1.tick_params(labelsize=10)
        ax1.set_title(f"R={correlation_coefficients:.5f}", fontsize=12)
        ax1.grid()

    @classmethod
    def __compute_correlations(cls, variable_history: NumberArray) -> RealArray:
        """Compute the correlations between the variables.

        Args:
            variable_history: The history of the variables.

        Returns:
            The lower diagonal of the correlation matrix of the variables.
        """
        return np.tril(
            atleast_2d(np.corrcoef(variable_history.astype(float), rowvar=False))
        )

    def __sort_variable_names(
        self,
        variable_names: Sequence[str],
        func_names: Sequence[str],
    ) -> list[str]:
        """Sort the expanded variable names using func_names as the pattern.

        In addition to sorting the expanded variable names, this method
        replaces the default hard-coded vectors (x_1, x_2, ... x_n) with
        the names given by the user.

        Args:
            variable_names: The expanded variable names to be sorted.
            func_names: The functions names in the required order.

        Returns:
            The sorted expanded variable names.
        """
        v_names = sorted(variable_names, key=partial(self.func_order, func_names))
        x_names = self._get_design_variable_names()
        return v_names[: -len(x_names)] + x_names

    @staticmethod
    def func_order(
        func_names: Sequence[str],
        x: str,
    ) -> tuple[int, str]:
        """Key function to sort function components.

        Args:
            func_names: The functions names in the required order.
            x: An element from a list.

        Returns:
            The index to be given to the sort method
            and the function name associated to that index.
        """
        for func_index, func_name in enumerate(func_names):
            # Escape special characters that may be present in the function name,
            # typically [, ], -
            regex_func_name = re.escape(func_name)
            if fullmatch(rf"{regex_func_name}((_\d+)|(\[\d\]))?", x):
                return func_index, x.replace(func_name, "")

        return len(func_names) + 1, x
