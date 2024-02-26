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
"""Plot the constraints on a radar chart at a given database index."""

from __future__ import annotations

from typing import ClassVar

from numpy import vstack
from numpy import zeros

from gemseo.datasets.dataset import Dataset
from gemseo.post.base_post import BasePost
from gemseo.post.dataset.radar_chart import RadarChart as RadarChartPost
from gemseo.post.radar_chart_settings import RadarChartSettings


class RadarChart(BasePost[RadarChartSettings]):
    """Plot the constraints on a radar chart at a given database index."""

    Settings: ClassVar[type[RadarChartSettings]] = RadarChartSettings

    def _plot(self, settings: RadarChartSettings) -> None:
        """
        Raises:
            ValueError: When a requested name is not a constraint
                or when the requested iteration is neither a database index
                nor the tag ``"opt"``.
        """  # noqa: D205, D212, D415
        constraint_names = settings.constraint_names
        iteration = settings.iteration

        if not constraint_names:
            constraint_names = self.optimzation_problem.get_constraint_names()
        else:
            constraint_names = self.optimzation_problem.get_function_names(constraint_names)
            invalid_names = sorted(
                set(constraint_names)
                - set(self.optimization_problem.constraints.get_names())
            )
            if invalid_names:
                msg = (
                    f"The names {invalid_names} are not names of constraints "
                    "stored in the database."
                )
                raise ValueError(msg)
        else:
            constraint_names = self.optimization_problem.constraints.get_names()

        # optimum_index is the zero-based position of the optimum.
        # while an iteration is a one-based position.
        assert self.opt_problem.solution is not None
        assert self.opt_problem.solution.optimum_index is not None
        optimum_iteration = self.optimzation_problem.solution.optimum_index + 1
        if iteration is None:
            iteration = optimum_iteration
        is_optimum = iteration == optimum_iteration

        n_iterations = len(self.database)
        if abs(iteration) not in range(1, n_iterations + 1):
            msg = (
                f"The requested iteration {iteration} is neither "
                f"in ({-n_iterations},...,-1,1,...,{n_iterations}) "
                f"nor None."
            )
            raise ValueError(msg)

        if iteration < 0:
            iteration = n_iterations + iteration + 1

        constraint_values, constraint_names, _ = self.database.get_history_array(
            function_names=constraint_names, with_x_vect=False
        )
        # "-1" because ndarray uses zero-based indexing and iteration is one-based.
        constraint_values = constraint_values[iteration - 1, :].ravel()

        dataset = Dataset(dataset_name="Constraints")
        values = vstack((constraint_values, zeros(len(constraint_values))))
        dataset.add_group(
            dataset.DEFAULT_GROUP,
            values,
            constraint_names,
        )
        dataset.index = ["computed constraints", "limit constraint"]  # type: ignore

        radar = RadarChartPost(
            dataset, display_zero=False, radial_ticks=settings.show_names_radially
        )
        radar.linestyle = ["-", "--"]
        radar.color = ["k", "r"]
        title_suffix = " (optimum)" if is_optimum else ""
        radar.title = f"Constraints at iteration {iteration}{title_suffix}"

        figures = radar.execute(save=False)
        for figure in figures:
            self._add_figure(figure)
