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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A heat map of a dataset, either rectangular or symmetric."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import ix_

from gemseo.post.dataset.base import BaseDatasetPlot
from gemseo.post.dataset.heatmap_settings import Heatmap_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Heatmap(BaseDatasetPlot[Heatmap_Settings]):
    """A heat map of a dataset, either rectangular or symmetric.

    When
    [Heatmap_Settings.symmetric][gemseo.post.dataset.heatmap_settings.Heatmap_Settings]
    is `False` (default),
    plot the evolution of the variables over the samples of the dataset,
    with rows as variables and columns as samples.

    When it is `True`,
    plot a symmetric variable-versus-variable matrix,
    e.g. a correlation matrix or second-order Sobol' indices.
    In that case,
    the contents of the rows in the dataset must match the contents of its columns.
    In other words,
    the underlying array must be symmetric.
    Or,
    equivalently,
    the `i`-th entry of the dataset must correspond to the `i`-th scalar variable.

    Note:
        When both `use_log` and `center` (in `symmetric` mode) are set,
        `use_log` takes precedence: the colormap is log-scaled and not centered.
    """

    settings_class = Heatmap_Settings

    def _create_specific_data_from_dataset(self) -> tuple[RealArray, tuple[str, ...]]:
        """
        Returns:
            The data to be plotted,
            the names of the variables.

        Raises:
            ValueError: If `symmetric` is `True`
                and the dataset does not define a square matrix,
                i.e. does not have as many samples as scalar variables.
        """  # noqa: D205, D212, D415
        settings = self.settings
        if settings.symmetric:
            all_variable_names = self.dataset.get_columns()
            data = self.dataset.get_view().to_numpy()
            variable_names = all_variable_names
            if settings.variables:
                variable_names = self.dataset.get_columns(
                    variable_names=settings.variables
                )
                indices = [all_variable_names.index(name) for name in variable_names]
                data = data[ix_(indices, indices)]

            n_rows, n_cols = data.shape
            if n_rows != n_cols:
                msg = (
                    "Heatmap with symmetric=True requires a square dataset, "
                    "i.e. as many samples as scalar variables; "
                    f"got {n_rows} samples and {n_cols} scalar variables."
                )
                raise ValueError(msg)

            return data, tuple(variable_names)

        return (
            self.dataset.get_view(variable_names=settings.variables).to_numpy().T,
            tuple(self.dataset.get_columns(variable_names=settings.variables)),
        )
