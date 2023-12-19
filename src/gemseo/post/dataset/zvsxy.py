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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Draw a variable versus two others from a :class:`.Dataset`.

A :class:`.ZvsXY` plot represents the variable :math:`z` with respect to
:math:`x` and :math:`y` as a surface plot, based on a set of points
:points :math:`\{x_i,y_i,z_i\}_{1\leq i \leq n}`. This interpolation is
relies on the Delaunay triangulation of :math:`\{x_i,y_i\}_{1\leq i \leq n}`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot
from gemseo.post.dataset.dataset_plot import VariableType

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from gemseo.datasets.dataset import Dataset


class ZvsXY(DatasetPlot):
    """Plot surface z versus x,y."""

    def __init__(
        self,
        dataset: Dataset,
        x: VariableType,
        y: VariableType,
        z: VariableType,
        add_points: bool = False,
        fill: bool = True,
        levels: int | Sequence[int] | None = None,
        other_datasets: Iterable[Dataset] | None = None,
    ) -> None:
        """
        Args:
            x: The name of the variable on the x-axis,
                with its optional component if not ``0``,
                e.g. ``("foo", 3)`` for the fourth component of the variable ``"foo"``.
            y: The name of the variable on the y-axis,
                with its optional component if not ``0``,
                e.g. ``("bar", 3)`` for the fourth component of the variable ``"bar"``.
            z: The name of the variable on the z-axis,
                with its optional component if not ``0``,
                e.g. ``("baz", 3)`` for the fourth component of the variable ``"baz"``.
            add_points: Whether to display the entries of the dataset as points
                above the surface.
            fill: Whether to generate a filled contour plot.
            levels: Either the number of contour lines
                or the values of the contour lines in increasing order.
                If ``None``, select them automatically.
            other_datasets: Additional datasets to be plotted as points
                above the surface.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset=dataset,
            x=self._force_variable_to_tuple(x),
            y=self._force_variable_to_tuple(y),
            z=self._force_variable_to_tuple(z),
            add_points=add_points,
            other_datasets=other_datasets,
            fill=fill,
            levels=levels,
        )

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[NDArray[float], NDArray[float], NDArray[float], Iterable[Dataset]]:
        """
        Returns:
            The values of the points on the x-axis,
            the values of the points on the y-axis,
            the values of the points on the z-axis,
            and possibly other datasets.
        """  # noqa: D205, D212, D415
        other_datasets = self._specific_settings.other_datasets or []
        self._n_items = 1 + len(other_datasets)
        self._set_color(self._n_items)
        x, x_comp = self._specific_settings.x
        y, y_comp = self._specific_settings.y
        z, z_comp = self._specific_settings.z
        self.xlabel = self.xlabel or self._get_component_name(
            x, x_comp, self.dataset.variable_names_to_n_components
        )
        self.ylabel = self.ylabel or self._get_component_name(
            y, y_comp, self.dataset.variable_names_to_n_components
        )
        self.zlabel = self.zlabel or self._get_component_name(
            z, z_comp, self.dataset.variable_names_to_n_components
        )
        self.title = self.title or self.zlabel
        get_view = self.dataset.get_view
        return (
            get_view(variable_names=x, components=x_comp).to_numpy().ravel(),
            get_view(variable_names=y, components=y_comp).to_numpy().ravel(),
            get_view(variable_names=z, components=z_comp).to_numpy().ravel(),
            other_datasets,
        )
