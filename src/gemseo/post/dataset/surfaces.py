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
"""Draw surfaces from a :class:`.Dataset`.

A :class:`.Surfaces` plot represents samples
of a functional variable :math:`z(x,y)` discretized over a 2D mesh.
Both evaluations of :math:`z` and mesh are stored in a :class:`.Dataset`,
:math:`z` as a parameter and the mesh as a metadata.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.datasets.dataset import Dataset


class Surfaces(DatasetPlot):
    """Plot surfaces y_i over the mesh x."""

    def __init__(
        self,
        dataset: Dataset,
        mesh: str,
        variable: str,
        samples: Sequence[int] | None = None,
        add_points: bool = False,
        fill: bool = True,
        levels: int | Sequence[int] | None = None,
    ) -> None:
        """
        Args:
            mesh: The name of the dataset metadata corresponding to the mesh.
            variable: The name of the variable for the x-axis.
            samples: The indices of the samples to plot. If ``None``, plot all samples.
            add_points: If ``True`` then display the samples over the surface plot.
            fill: Whether to generate a filled contour plot.
            levels: Either the number of contour lines
                or the values of the contour lines in increasing order.
                If ``None``, select them automatically.
        """  # noqa: D205, D212, D415
        super().__init__(
            dataset,
            mesh=mesh,
            variable=variable,
            samples=samples,
            add_points=add_points,
            fill=fill,
            levels=levels,
        )
