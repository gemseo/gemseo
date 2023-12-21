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
"""Evolution of the variables by means of a color scale."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray

    from gemseo.datasets.dataset import Dataset


class ColorEvolution(DatasetPlot):
    """Evolution of the variables by means of a color scale.

    Based on the matplotlib function :meth:`imshow`.

    Tip:
        Use :attr:`.colormap` to set a matplotlib colormap, e.g. ``"seismic"``.
    """

    def __init__(
        self,
        dataset: Dataset,
        variables: Iterable[str] | None = None,
        use_log: bool = False,
        opacity: float = 0.6,
        **options: bool | float | str | None,
    ) -> None:
        """
        Args:
            variables: The variables of interest
                If ``None``, use all the variables.
            use_log: Whether to use a symmetric logarithmic scale.
            opacity: The level of opacity (0 = transparent; 1 = opaque).
            **options: The options for the matplotlib function :meth:`imshow`.
        """  # noqa: D205, D212, D415
        options_ = {
            "interpolation": "nearest",
            "aspect": "auto",
        }
        options_.update(options)
        super().__init__(
            dataset,
            variables=variables,
            use_log=use_log,
            opacity=opacity,
            options=options_,
        )

    def _create_specific_data_from_dataset(self) -> tuple[NDArray[float], list[str]]:
        """
        Returns:
            The data to be plotted,
            the names of the variables.
        """  # noqa: D205, D212, D415
        variable_names = (
            self._specific_settings.variables or self.dataset.variable_names
        )
        return (
            self.dataset.get_view(variable_names=variable_names).to_numpy().T,
            variable_names,
        )
