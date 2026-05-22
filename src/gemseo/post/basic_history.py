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

import operator
from typing import ClassVar
from typing import Final

from numpy import arange
from numpy import newaxis

from gemseo.post.base_post import BasePost
from gemseo.post.basic_history_settings import BasicHistory_Settings
from gemseo.post.dataset.lines import Lines
from gemseo.post.dataset.lines_settings import Lines_Settings


class BasicHistory(BasePost[BasicHistory_Settings]):
    """Plot the history of selected constraint, objective and observable functions.

    This post-processor requires the names of these selected outputs.
    """

    settings_class: ClassVar[type[BasicHistory_Settings]] = BasicHistory_Settings

    __ITERATION_NAME: Final[str] = ",;:!"
    """The name for the variable iteration in the dataset.

    A name that a user cannot chose for its own variables. Only used in the background.
    """

    def _plot(self, settings: BasicHistory_Settings) -> None:  # noqa: D205, D212, D415
        optimization_metadata = self._optimization_metadata
        dataset = self._dataset.copy()
        dataset.add_variable(
            self.__ITERATION_NAME, arange(1, len(dataset) + 1)[:, newaxis]
        )

        variable_names = list(settings.variable_names)
        if optimization_metadata.objective_name in variable_names:
            if (
                optimization_metadata.use_standardized_objective
                and not optimization_metadata.minimize_objective
            ):
                obj_index = variable_names.index(optimization_metadata.objective_name)
                variable_names[obj_index] = (
                    optimization_metadata.standardized_objective_name
                )

            if self._change_obj:
                dataset.transform_data(
                    operator.neg,
                    variable_names=optimization_metadata.standardized_objective_name,
                )
                dataset.rename_variable(
                    optimization_metadata.standardized_objective_name,
                    optimization_metadata.objective_name,
                )

        if settings.normalize:
            dataset = dataset.get_normalized()

        settings = Lines_Settings(
            abscissa_variable=self.__ITERATION_NAME,
            variables=tuple(
                names
                for variable_name in variable_names
                for names in (
                    optimization_metadata.output_name_to_constraint_names.get(
                        variable_name, [variable_name]
                    )
                )
            ),
            use_integer_xticks=True,
            font_size=12,
            xlabel="Iterations",
            title="History plot",
            fig_size=settings.fig_size,
        )
        plot = Lines(dataset, settings)
        self._add_figure(plot)
