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
"""Plot the constraints on a radar chart at a given dataset index."""

from __future__ import annotations

from typing import ClassVar

from numpy import vstack
from numpy import zeros

from gemseo.datasets.dataset import Dataset
from gemseo.post.base_post import BasePost
from gemseo.post.dataset.radar_chart import RadarChart as RadarChartPost
from gemseo.post.radar_chart_settings import RadarChart_Settings


class RadarChart(BasePost[RadarChart_Settings]):
    """Plot the constraints on a radar chart at a given dataset index."""

    Settings: ClassVar[type[RadarChart_Settings]] = RadarChart_Settings

    def _plot(self, settings: RadarChart_Settings) -> None:
        """
        Raises:
            ValueError: When a requested name is not a constraint
                or when the requested iteration is neither a dataset index
                nor the tag ``"opt"``.
        """  # noqa: D205, D212, D415
        constraint_names = settings.constraint_names
        constraint_names = (
            list(constraint_names)
            if not isinstance(constraint_names, list)
            else constraint_names
        )
        iteration = settings.iteration
        constraint_names_mapping = (
            self._optimization_metadata.output_names_to_constraint_names
        )

        if constraint_names:
            names = []
            for constraint_name in constraint_names:
                if constraint_name in constraint_names_mapping:
                    names.extend(constraint_names_mapping[constraint_name])
                else:
                    names.append(constraint_name)
            constraint_names = names
            invalid_names = sorted(
                set(constraint_names)
                - set(
                    self._dataset.inequality_constraint_names
                    + self._dataset.equality_constraint_names
                )
            )
            if invalid_names:
                msg = (
                    f"The names {invalid_names} are not names of constraints "
                    "stored in the dataset."
                )
                raise ValueError(msg)
        else:
            constraint_names = (
                self._dataset.equality_constraint_names
                + self._dataset.inequality_constraint_names
            )
        # optimum_index is the zero-based position of the optimum.
        # while an iteration is a one-based position.
        assert self._optimization_metadata.optimum_iteration is not None
        optimum_iteration = self._optimization_metadata.optimum_iteration
        if iteration is None:
            iteration = optimum_iteration
        is_optimum = iteration == optimum_iteration
        dataset = self._dataset
        n_iterations = len(dataset)
        if abs(iteration) not in range(1, n_iterations + 1):
            msg = (
                f"The requested iteration {iteration} is neither "
                f"in ({-n_iterations},...,-1,1,...,{n_iterations}) "
                f"nor None."
            )
            raise ValueError(msg)

        if iteration < 0:
            iteration = n_iterations + iteration + 1

        constraint_values = dataset.get_view(variable_names=constraint_names).to_numpy()
        constraint_names = dataset.get_columns(constraint_names)

        # "-1" because ndarray uses zero-based indexing and iteration is one-based.
        constraint_values = constraint_values[iteration - 1, :].ravel()

        dataset = Dataset(dataset_name="Constraints")
        values = vstack((constraint_values, zeros(len(constraint_values))))
        dataset.add_group(
            dataset.DEFAULT_GROUP,
            values,
            constraint_names,
        )
        dataset.index = ["computed constraints", "limit constraint"]

        radar = RadarChartPost(
            dataset, display_zero=False, radial_ticks=settings.show_names_radially
        )
        radar.linestyle = ["-", "--"]
        radar.color = ["k", "r"]
        title_suffix = " (optimum)" if is_optimum else ""
        radar.title = f"Constraints at iteration {iteration}{title_suffix}"
        self._add_figure(radar)
