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
from typing import TYPE_CHECKING
from typing import ClassVar

import numpy as np
from matplotlib import pyplot
from numpy import arange
from numpy import atleast_2d
from numpy import where

from gemseo.algos.database import Database
from gemseo.post.base_post import BasePost
from gemseo.post.gradient_sensitivity_settings import GradientSensitivity_Settings
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.figure import Figure
    from numpy import ndarray

    from gemseo.typing import NumberArray
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


class GradientSensitivity(BasePost[GradientSensitivity_Settings]):
    """Derivatives of the objective and constraints at a given iteration."""

    Settings: ClassVar[type[GradientSensitivity_Settings]] = (
        GradientSensitivity_Settings
    )

    _USE_JACOBIAN_DATA: ClassVar[bool] = True

    def _plot(self, settings: GradientSensitivity_Settings) -> None:
        compute_missing_gradients = settings.compute_missing_gradients

        if settings.iteration is None:
            design_value = self._dataset.design_dataset.loc[
                self._optimization_metadata.optimum_iteration
            ].values
        else:
            design_value = self._dataset.design_dataset.loc[
                settings.iteration
            ].to_numpy()
        fig = self.__generate_subplots(
            self._get_design_variable_names(),
            design_value,
            self.__get_output_gradients(
                design_value,
                iteration=settings.iteration,
                scale_gradients=settings.scale_gradients,
                compute_missing_gradients=compute_missing_gradients,
            ),
            settings.scale_gradients,
            settings.fig_size,
        )
        fig.tight_layout()
        self._add_figure(fig)

    def __get_output_gradients(
        self,
        design_value: ndarray,
        iteration: int | None = None,
        scale_gradients: bool = False,
        compute_missing_gradients: bool = False,
    ) -> dict[str, RealArray]:
        """Return the gradients of all the output variable at a given design value.

        Args:
            design_value: The value of the design vector.
            scale_gradients: Whether to normalize the gradients
                w.r.t. the design variables.
            compute_missing_gradients: Whether to compute the gradients at the
                selected iteration if they were not computed by the algorithm.

                .. warning::
                   Activating this option may add considerable computation time
                   depending on the cost of the gradient evaluation.
                   This option will not compute the gradients if the
                   :class:`.OptimizationProblem` instance was imported from an HDF5
                   file. This option requires an :class:`.OptimizationProblem` with a
                   gradient-based algorithm.

        Returns:
            The gradients of the outputs
            indexed by the names of the output,
            e.g. ``"output_name"`` for a mono-dimensional output,
            or ``"output_name_i"`` for the i-th component of a multidimensional output.
        """
        gradient_values = {}
        if compute_missing_gradients:
            try:
                output_functions, jacobian_functions = (
                    self.optimization_problem.get_functions(
                        no_db_no_norm=True, jacobian_names=()
                    )
                )
                _, gradient_values = self.optimization_problem.evaluate_functions(
                    design_vector=design_value,
                    design_vector_is_normalized=False,
                    output_functions=output_functions or None,
                    jacobian_functions=jacobian_functions or None,
                )
            except NotImplementedError:
                LOGGER.info(
                    "The missing gradients for an OptimizationProblem without "
                    "callable functions cannot be computed."
                )

        function_names = (
            self._dataset.equality_constraint_names
            + self._dataset.inequality_constraint_names
            + self._dataset.objective_names
            + self._dataset.observable_names
        )
        scale_gradient = self._dataset.misc["input_space"].unnormalize_vect
        function_names_to_gradients = {}
        for function_name in function_names:
            if compute_missing_gradients and gradient_values:
                gradient_value = gradient_values[function_name]
            else:
                grad_name = Database.get_gradient_name(function_name)
                if iteration is None:
                    iteration = self._optimization_metadata.optimum_iteration
                try:
                    gradient_value = (
                        self._dataset.get_view(variable_names=grad_name)
                        .loc[iteration]
                        .to_numpy()
                    )
                    if not np.isnan(gradient_value).any():
                        if gradient_value.shape != design_value.shape:
                            gradient_value = gradient_value.reshape(
                                -1, len(design_value)
                            )
                    else:
                        gradient_value = None
                        continue
                except KeyError:
                    gradient_value = None

            if gradient_value is None:
                continue

            gradient_value = atleast_2d(gradient_value)
            size = len(gradient_value)
            for i, gradient_value_ in enumerate(gradient_value):
                if scale_gradients:
                    gradient_value_ = scale_gradient(gradient_value_, minus_lb=False)
                function_names_to_gradients[repr_variable(function_name, i, size)] = (
                    gradient_value_
                )

        return function_names_to_gradients

    def __generate_subplots(
        self,
        design_names: Iterable[str],
        design_value: NumberArray,
        gradients: dict[str, RealArray],
        scale_gradients: bool,
        fig_size: tuple[float, float],
    ) -> Figure:
        """Generate the gradients subplots from the data.

        Args:
            design_names: The names of the design variables.
            design_value: The reference value for x.
            gradients: The gradients to plot indexed by the output names.
            scale_gradients: Whether to normalize the gradients
                w.r.t. the design variables.
            fig_size: The size of the figure.

        Returns:
            The gradients subplots.

        Raises:
            ValueError: If `gradients` is empty.
        """
        n_gradients = len(gradients)
        if n_gradients == 0:
            msg = "No gradients to plot at current iteration."
            raise ValueError(msg)

        n_cols = 2
        n_rows = sum(divmod(n_gradients, n_cols))

        fig, axs = pyplot.subplots(
            nrows=n_rows, ncols=n_cols, sharex=True, figsize=fig_size, squeeze=False
        )

        abscissa = arange(len(design_value))
        if self._change_obj:
            gradients[self._optimization_metadata.objective_name] = -gradients.pop(
                self._optimization_metadata.standardized_objective_name
            )

        i = j = 0
        font_size = 12
        rotation = 90
        for output_name, gradient_value in sorted(gradients.items()):
            ax = axs[i][j]
            ax.bar(
                abscissa,
                gradient_value,
                color=where(gradient_value < 0, "blue", "red"),
                align="center",
            )
            ax.grid()
            ax.set_axisbelow(True)
            ax.set_title(output_name)
            ax.set_xticks(abscissa)
            ax.set_xticklabels(design_names, fontsize=font_size, rotation=rotation)
            # Update y labels spacing
            vis_labels = [
                label for label in ax.get_yticklabels() if label.get_visible() is True
            ]
            pyplot.setp(vis_labels[::2], visible=False)
            if j == n_cols - 1:
                j = 0
                i += 1
            else:
                j += 1

        if j == n_cols - 1:
            ax = axs[i][j]
            ax.set_xticks(abscissa)
            ax.set_xticklabels(design_names, fontsize=font_size, rotation=rotation)

        title = (
            "Derivatives of objective and constraints with respect to design variables"
        )
        if scale_gradients:
            fig.suptitle(f"{title}\n\nNormalized design space.", fontsize=14)
        else:
            fig.suptitle(title, fontsize=14)
        return fig
