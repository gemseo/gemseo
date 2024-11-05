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
"""Plot the partial sensitivity of the functions."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar

from matplotlib import pyplot
from numpy import absolute
from numpy import argsort
from numpy import array
from numpy import savetxt
from numpy import stack

from gemseo.post.base_post import BasePost
from gemseo.post.variable_influence_settings import VariableInfluence_Settings
from gemseo.utils.string_tools import pretty_str
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matplotlib.figure import Figure

    from gemseo.typing import RealArray


LOGGER = logging.getLogger(__name__)


class VariableInfluence(BasePost[VariableInfluence_Settings]):
    r"""First order variable influence analysis.

    This post-processing computes
    :math:`\frac{\partial f(x)}{\partial x_i}\left(x_i^* - x_i^{(0)}\right)`
    where :math:`x_i^{(0)}` is the initial value of the variable
    and :math:`x_i^*` is the optimal value of the variable.

    Options of the plot method are:

    - proportion of the total sensitivity
      to use as a threshold to filter the variables,
    - the use of a logarithmic scale,
    - the possibility to save the indices of the influential variables indices
      in a NumPy file.
    """

    Settings: ClassVar[type[VariableInfluence_Settings]] = VariableInfluence_Settings

    def _plot(self, settings: VariableInfluence_Settings) -> None:
        level = settings.level
        absolute_value = settings.absolute_value
        log_scale = settings.log_scale
        save_var_files = settings.save_var_files

        function_names = self.optimization_problem.function_names
        _, x_opt, _, _, _ = self.optimization_problem.optimum
        x_0 = self.database.get_x_vect(1)
        absolute_value = log_scale or absolute_value

        names_to_sensitivities = {}
        evaluate = self.database.get_function_value
        for function_name in function_names:
            grad = evaluate(self.database.get_gradient_name(function_name), x_0)
            if grad is None:
                continue

            f_0 = evaluate(function_name, x_0)
            f_opt = evaluate(function_name, x_opt)
            if self._change_obj and function_name == self._neg_obj_name:
                grad = -grad
                function_name = self._obj_name

            if len(grad.shape) == 1:
                sensitivity = grad * (x_opt - x_0)
                sensitivity *= (f_opt - f_0) / sensitivity.sum()
                if absolute_value:
                    sensitivity = absolute(sensitivity)
                names_to_sensitivities[function_name] = sensitivity
            else:
                for i, _grad in enumerate(grad):
                    sensitivity = _grad * (x_opt - x_0)
                    sensitivity *= (f_opt - f_0)[i] / sensitivity.sum()
                    if absolute_value:
                        sensitivity = absolute(sensitivity)
                    names_to_sensitivities[repr_variable(function_name, i)] = (
                        sensitivity
                    )

        self._add_figure(
            self.__generate_subplots(
                names_to_sensitivities,
                level,
                log_scale,
                save_var_files,
                settings.fig_size,
            )
        )

    def __get_quantile(
        self,
        sensitivity: RealArray,
        func: str,
        level: float = 0.99,
        save: bool = False,
    ) -> tuple[int, float]:
        """Get the number of variables explaining a fraction of the sensitivity.

        Args:
            sensitivity: The sensitivity.
            func: The function name.
            level: The quantile level.
            save: Whether to save the influential variables indices in a NumPy file.

        Returns:
            The number of influential variables
            and the absolute sensitivity w.r.t. the least influential variable.
        """
        absolute_sensitivity = absolute(sensitivity)
        absolute_sensitivity_indices = argsort(absolute_sensitivity)[::-1]
        absolute_sensitivity = absolute_sensitivity[absolute_sensitivity_indices]
        variance = 0.0
        total_variance = absolute_sensitivity.sum() * level
        n_variables = 0
        while variance < total_variance and n_variables < len(absolute_sensitivity):
            variance += absolute_sensitivity[n_variables]
            n_variables += 1

        influential_variables = absolute_sensitivity_indices[:n_variables]
        x_names = self._get_design_variable_names()
        LOGGER.info(
            "   %s; %s",
            func,
            pretty_str([x_names[i] for i in influential_variables]),
        )
        if save:
            names = [
                [
                    f"{name}${i}"
                    for i in range(
                        self.optimization_problem.design_space.get_size(name)
                    )
                ]
                for name in self.optimization_problem.design_space  # noqa: E501
            ]
            names = array(list(itertools.chain(*names)))
            file_name = f"{func}_influ_vars.csv"
            savetxt(
                file_name,
                stack((names[influential_variables], influential_variables)).T,
                fmt="%s",
                delimiter=" ; ",
                header="name ; index",
            )
            self._output_file_paths.append(Path(file_name))

        return n_variables, absolute_sensitivity[n_variables - 1]

    def __generate_subplots(
        self,
        names_to_sensitivities: Mapping[str, RealArray],
        level: float,
        log_scale: bool,
        save: bool,
        fig_size: tuple[float, float],
    ) -> Figure:
        """Generate the gradients subplots from the data.

        Args:
            names_to_sensitivities: The output sensitivities
                w.r.t. the design variables.
            level: The proportion of the total sensitivity
                to use as a threshold to filter the variables.
            log_scale: Whether to set the y-axis as log scale.
            save: Whether to save the influential variables indices in a NumPy file.

        Returns:
            The gradients subplots.

        Raises:
            ValueError: If the `names_to_sensitivities` is empty.
        """
        n_funcs = len(names_to_sensitivities)
        if not n_funcs:
            msg = "No gradients to plot at current iteration."
            raise ValueError(msg)

        n_rows, n_cols = self.__compute_optimal_grid(n_funcs)
        fig, axs = pyplot.subplots(
            nrows=n_rows, ncols=n_cols, sharex=True, figsize=fig_size
        )

        x_labels = self._get_design_variable_names()
        # This variable determines the number of variables to plot in the
        # x-axis. Since the data history can be edited by the user after the
        # problem was solved, we do not use something like opt_problem.dimension
        # because the problem dimension is not updated when the history is filtered.
        abscissas = range(len(next(iter(names_to_sensitivities.values()))))

        LOGGER.info(
            "Output name; "
            "most influential variables to explain %s%% of the output variation ",
            level,
        )
        for index, (name, sensitivity) in enumerate(
            sorted(names_to_sensitivities.items())
        ):
            i = index // n_cols
            j = index % n_cols
            ax = axs[i][j]
            quantile, threshold = self.__get_quantile(
                sensitivity, name, level=level, save=save
            )
            ax.fill_between(
                [-1, len(sensitivity) + 1],
                -threshold,
                threshold,
                color="gray",
                facecolor="none",
                hatch="///",
                label="Non-influential domain",
            )
            ax.axhline(y=0.0, color="black")
            ax.bar(
                abscissas,
                sensitivity,
                color="blue",
                align="center",
                label="Partial derivatives",
            )
            ax.set_title(
                f"{quantile} variables explain {round(level * 100)}% of {name}"
            )
            ax.set_xticks(abscissas)
            ax.set_xticklabels(x_labels, rotation=90)
            ax.set_xlim(-1, len(sensitivity) + 1)
            ax.grid()
            ax.set_axisbelow(True)
            if log_scale:
                ax.set_yscale("log")

            # Update y labels spacing
            vis_labels = [
                label for label in ax.get_yticklabels() if label.get_visible() is True
            ]
            pyplot.setp(vis_labels, visible=False)
            pyplot.setp(vis_labels[::2], visible=True)

            vis_xlabels = [
                label for label in ax.get_xticklabels() if label.get_visible() is True
            ]
            if len(vis_xlabels) > 20:
                pyplot.setp(vis_xlabels, visible=False)
                pyplot.setp(vis_xlabels[:: int(len(vis_xlabels) / 10.0)], visible=True)

        axs[0, 0].legend()
        fig.tight_layout()
        return fig

    @staticmethod
    def __compute_optimal_grid(n_items: int) -> tuple[int, int]:
        """Compute the optimal grid given a number of items.

        Args:
            n_items: The number of items.

        Returns:
            The optimal number of rows and columns.
        """
        optimal_n_rows, optimal_n_cols = 1, 1
        smallest_difference = float("inf")
        for n_rows in range(1, n_items + 1):
            for n_cols in range(1, n_items + 1):
                d1 = n_rows - n_cols
                d2 = n_rows * n_cols - n_items
                difference = (d1 + 1) * (d2 + 1)
                if d1 > 0 and d2 > 0 and difference < smallest_difference:
                    smallest_difference = difference
                    optimal_n_rows, optimal_n_cols = n_rows, n_cols

        return optimal_n_rows, optimal_n_cols
