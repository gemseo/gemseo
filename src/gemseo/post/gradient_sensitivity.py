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
"""Plot the derivatives of the functions."""
from __future__ import annotations

import logging
from typing import Iterable
from typing import Mapping

from matplotlib import pyplot
from matplotlib.figure import Figure
from numpy import arange
from numpy import atleast_2d
from numpy import ndarray

from gemseo.post.opt_post_processor import OptPostProcessor

LOGGER = logging.getLogger(__name__)


class GradientSensitivity(OptPostProcessor):
    """Derivatives of the objective and constraints at a given iteration."""

    DEFAULT_FIG_SIZE = (10.0, 10.0)

    def _plot(
        self,
        iteration: int | None = None,
        scale_gradients: bool = False,
    ) -> None:
        """
        Args:
            iteration: The iteration to plot the sensitivities.
                Can use either positive or negative indexing,
                e.g. ``4`` for the 5-th iteration
                or ``-2`` for the penultimate one.
                If ``None``, use the iteration of the optimum.
            scale_gradients: If True, normalize each gradient
                w.r.t. the design variables.
        """
        if iteration is None:
            design_value = self.opt_problem.solution.x_opt
        else:
            design_value = self.opt_problem.database.get_x_by_iter(iteration)

        fig = self.__generate_subplots(
            self._generate_x_names(),
            design_value,
            self.__get_output_gradients(design_value, scale_gradients=scale_gradients),
            scale_gradients=scale_gradients,
        )
        self._add_figure(fig)

    def __get_output_gradients(
        self,
        design_value: ndarray,
        scale_gradients: bool = False,
    ) -> dict[str, ndarray]:
        """Return the gradients of all the output variable at a given design value.

        Args:
            design_value: The value of the design vector.
            scale_gradients: Whether to normalize the gradients
                w.r.t. the design variables.

        Returns:
            The gradients of the outputs
            indexed by the names of the output,
            e.g. ``"output_name"`` for a mono-dimensional output,
            or ``"output_name_i"`` for the i-th component of a multi-dimensional output.
        """
        function_names = self.opt_problem.get_all_functions_names()
        scale_gradient = self.opt_problem.design_space.unnormalize_vect
        function_names_to_gradients = {}
        for function_name in function_names:
            gradient_value = self.database.get_f_of_x(f"@{function_name}", design_value)
            if gradient_value is None:
                continue

            if gradient_value.ndim == 1:
                if scale_gradients:
                    gradient_value = scale_gradient(gradient_value, minus_lb=False)
                function_names_to_gradients[function_name] = gradient_value
                continue

            for i, _gradient_value in enumerate(gradient_value):
                if scale_gradients:
                    _gradient_value = scale_gradient(_gradient_value, minus_lb=False)
                function_names_to_gradients[f"{function_name}_{i}"] = _gradient_value

        return function_names_to_gradients

    def __generate_subplots(
        self,
        design_names: Iterable[str],
        design_value: ndarray,
        gradients: Mapping[str, ndarray],
        scale_gradients: bool = False,
    ) -> Figure:
        """Generate the gradients subplots from the data.

        Args:
            design_names: The names of the design variables.
            design_value: The reference value for x.
            gradients: The gradients to plot indexed by the output names.
            scale_gradients: Whether to normalize the gradients
                w.r.t. the design variables.

        Returns:
            The gradients subplots.

        Raises:
            ValueError: If `gradients` is empty.
        """
        n_gradients = len(gradients)
        if n_gradients == 0:
            raise ValueError("No gradients to plot at current iteration!")

        n_cols = 2
        n_rows = sum(divmod(n_gradients, n_cols))

        fig, axes = pyplot.subplots(
            nrows=n_rows,
            ncols=n_cols,
            sharex=True,
            sharey=False,
            figsize=self.DEFAULT_FIG_SIZE,
        )

        axes = atleast_2d(axes)
        abscissa = arange(len(design_value))
        if self._change_obj:
            gradients[self._obj_name] = -gradients.pop(self._standardized_obj_name)

        i = j = 0
        font_size = 12
        rotation = 90
        for output_name, gradient_value in sorted(gradients.items()):
            axe = axes[i][j]
            axe.bar(abscissa, gradient_value, color="blue", align="center")
            axe.set_title(output_name)
            axe.set_xticklabels(design_names, fontsize=font_size, rotation=rotation)
            axe.set_xticks(abscissa)
            # Update y labels spacing
            vis_labels = [
                label for label in axe.get_yticklabels() if label.get_visible() is True
            ]
            pyplot.setp(vis_labels[::2], visible=False)
            if j == n_cols - 1:
                j = 0
                i += 1
            else:
                j += 1

        if j == n_cols - 1:
            axe = axes[i][j]
            axe.set_xticklabels(design_names, fontsize=font_size, rotation=rotation)
            axe.set_xticks(abscissa)

        title = (
            "Derivatives of objective and constraints with respect to design variables"
        )
        if scale_gradients:
            fig.suptitle(f"{title}\n\nNormalized design space.", fontsize=14)
        else:
            fig.suptitle(title, fontsize=14)
        return fig
