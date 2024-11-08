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
#                         documentation
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A 1D Jameson sensor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import abs as np_abs
from numpy import amax

from gemseo.mlearning.transformers.base_transformer import BaseTransformer
from gemseo.mlearning.transformers.base_transformer import TransformerFitOptionType

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class JamesonSensor(BaseTransformer):
    """A 1D Jameson Sensor."""

    def __init__(
        self,
        name: str = "",
        threshold: float = 0.3,
        removing_part: float = 0.01,
        dimension: int = 1,
    ) -> None:
        """
        Args:
            name: A name for this transformer.
            threshold: The value to add to the denominator
                to avoid zero division.
            removing_part: The level of the signal to
                remove in order to avoid leading and trailing edge effects.
            dimension: The dimension of the mesh.
        """  # noqa: D205 D212
        super().__init__(name)
        self.threshold = threshold
        self.removing_part = removing_part
        self.dimension = dimension

    def _fit(self, data: RealArray, *args: TransformerFitOptionType) -> None:
        self.threshold *= amax(data)

    @BaseTransformer._use_2d_array
    def transform(  # noqa: D102
        self,
        data: RealArray,
    ) -> RealArray:
        mesh_size = data.shape[1] - 2
        min_mesh_size = int(mesh_size * self.removing_part)
        max_mesh_size = int(mesh_size * (1 - self.removing_part))
        norm = (
            np_abs(data[:, :-2])
            + 2 * np_abs(data[:, 1:-1])
            + np_abs(data[:, 2:])
            + self.threshold
        )
        result = abs(data[:, :-2] - 2 * data[:, 1:-1] + data[:, 2:]) / norm
        return result[:, min_mesh_size:max_mesh_size]
