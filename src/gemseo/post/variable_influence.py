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
from typing import Mapping

from matplotlib import pyplot
from matplotlib.figure import Figure
from numpy import absolute
from numpy import argsort
from numpy import array
from numpy import atleast_2d
from numpy import ndarray
from numpy import savetxt
from numpy import stack

from gemseo.post.opt_post_processor import OptPostProcessor

LOGGER = logging.getLogger(__name__)


class VariableInfluence(OptPostProcessor):
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

    DEFAULT_FIG_SIZE = (20.0, 5.0)

    def _plot(
        self,
        level: float = 0.99,
        absolute_value: bool = False,
        log_scale: bool = False,
        save_var_files: bool = False,
    ) -> None:
        """
        Args:
            level: The proportion of the total sensitivity
                to use as a threshold to filter the variables.
            absolute_value: Whether to plot the absolute value of the influence.
            log_scale: Whether to set the y-axis as log scale.
            save_var_files: Whether to save the influential variables indices
                to a NumPy file.
        """  # noqa: D205, D212, D415
        function_names = self.opt_problem.get_all_functions_names()
        _, x_opt, _, _, _ = self.opt_problem.get_optimum()
        x_0 = self.database.get_x_by_iter(0)
        absolute_value = log_scale or absolute_value

        names_to_sensitivities = {}
        evaluate = self.database.get_f_of_x
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
                    names_to_sensitivities[f"{function_name}_{i}"] = sensitivity

        self._add_figure(
            self.__generate_subplots(
                names_to_sensitivities,
                level=level,
                log_scale=log_scale,
                save=save_var_files,
            )
        )

    def __get_quantile(
        self,
        sensitivity: ndarray,
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
        LOGGER.info("VariableInfluence for function %s", func)
        LOGGER.info(
            "Most influential variables indices to explain "
            "%% of the function variation: %s",
            int(level * 100),
        )
        LOGGER.info(influential_variables)
        if save:
            names = [
                [f"{name}${i}" for i in range(size)]
                for name, size in self.opt_problem.design_space.variables_sizes.items()
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
            self.output_files.append(file_name)

        return n_variables, absolute_sensitivity[n_variables - 1]

    def __generate_subplots(
        self,
        names_to_sensitivities: Mapping[str, ndarray],
        level: float = 0.99,
        log_scale: bool = False,
        save: bool = False,
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
            raise ValueError("No gradients to plot at current iteration.")

        n_cols = 2
        n_rows = sum(divmod(n_funcs, n_cols))
        if n_funcs == 1:
            n_cols = 1

        fig, axes = pyplot.subplots(
            nrows=n_rows, ncols=n_cols, sharex=True, figsize=self.DEFAULT_FIG_SIZE
        )

        axes = atleast_2d(axes)
        x_labels = self._generate_x_names()
        # This variable determines the number of variables to plot in the
        # x-axis. Since the data history can be edited by the user after the
        # problem was solved, we do not use something like opt_problem.dimension
        # because the problem dimension is not updated when the history is filtered.
        abscissas = range(len(tuple(names_to_sensitivities.values())[0]))

        font_size = 12
        rotation = 90
        i = j = 0
        for name, sensitivity in sorted(names_to_sensitivities.items()):
            axe = axes[i][j]
            axe.bar(abscissas, sensitivity, color="blue", align="center")
            quantile, threshold = self.__get_quantile(
                sensitivity, name, level=level, save=save
            )
            axe.set_title(
                f"{quantile} variables required "
                f"to explain {round(level * 100)}% of {name} variations"
            )
            axe.set_xticklabels(x_labels, fontsize=font_size, rotation=rotation)
            axe.set_xticks(abscissas)
            axe.set_xlim(-1, len(sensitivity) + 1)
            axe.axhline(threshold, color="r")
            axe.axhline(-threshold, color="r")
            if log_scale:
                axe.set_yscale("log")

            # Update y labels spacing
            vis_labels = [
                label for label in axe.get_yticklabels() if label.get_visible() is True
            ]
            pyplot.setp(vis_labels, visible=False)
            pyplot.setp(vis_labels[::2], visible=True)

            vis_xlabels = [
                label for label in axe.get_xticklabels() if label.get_visible() is True
            ]
            if len(vis_xlabels) > 20:
                pyplot.setp(vis_xlabels, visible=False)
                pyplot.setp(vis_xlabels[:: int(len(vis_xlabels) / 10.0)], visible=True)

            if j == n_cols - 1:
                j = 0
                i += 1
            else:
                j += 1

        if len(names_to_sensitivities) < n_rows * n_cols:
            axe = axes[i][j]
            axe.set_xticklabels(x_labels, fontsize=font_size, rotation=rotation)
            axe.set_xticks(abscissas)

        fig.suptitle(
            "Partial variation of the functions wrt design variables", fontsize=14
        )
        return fig
