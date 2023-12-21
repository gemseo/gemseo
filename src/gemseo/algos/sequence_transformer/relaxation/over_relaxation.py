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
#        :author: Sebastien Bocquet, Alexandre Scotto Di Perrotolo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The over-relaxation method."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.sequence_transformer.sequence_transformer import SequenceTransformer

if TYPE_CHECKING:
    from typing import ClassVar

    from numpy.typing import NDArray


class OverRelaxation(SequenceTransformer):
    """The over relaxation method."""

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 2
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 0

    def __init__(self, factor: float = 1.0) -> None:
        """
        Args:
            factor: The relaxation factor lying within ]0, 2].

        Raises:
            ValueError if the provided relaxation factor lies outside ]0, 2].
        """  # noqa:D205 D212 D415
        super().__init__()

        self.factor = factor

    @property
    def factor(self) -> float:
        """The over-relaxation factor."""
        return self.__factor

    @factor.setter
    def factor(self, factor: float) -> None:
        if not (0 < factor <= 2):
            raise ValueError("Relax factor must lie within ]0, 2].")

        self.__factor = factor
        self.clear()

    def _compute_transformed_iterate(self) -> NDArray:
        gxn_1, gxn = self._iterates

        if self.__factor == 1.0:
            return gxn

        return self.__factor * gxn + (1.0 - self.__factor) * gxn_1
