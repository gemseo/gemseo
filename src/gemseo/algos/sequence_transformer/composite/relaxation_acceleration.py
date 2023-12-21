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

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.algos.sequence_transformer.composite.composite import (
    CompositeSequenceTransformer,
)
from gemseo.algos.sequence_transformer.relaxation.over_relaxation import OverRelaxation
from gemseo.algos.sequence_transformer.sequence_transformer_factory import (
    SequenceTransformerFactory,
)

if TYPE_CHECKING:
    from gemseo.algos.sequence_transformer.sequence_transformer import (
        SequenceTransformer,
    )


class RelaxationAcceleration(CompositeSequenceTransformer):
    """A composite made up of a relaxation followed by an acceleration."""

    _sequence_transformers: list[OverRelaxation | SequenceTransformer]
    """The sequence transformers that are chained."""

    __acceleration_method: AccelerationMethod
    """The acceleration method."""

    def __init__(
        self,
        over_relaxation_factor: float = 1.0,
        acceleration_method: AccelerationMethod = AccelerationMethod.NONE,
    ):
        """
        Args:
            over_relaxation_factor: The over-relaxation factor.
            acceleration_method: The acceleration method to be used to improve the
                convergence rate of the fixed point iteration method.
        """  # noqa:D205 D212 D415
        self.__sequence_transformer_factory = SequenceTransformerFactory()

        _sequence_transformers = [
            OverRelaxation(over_relaxation_factor),
            self.__sequence_transformer_factory.create(acceleration_method),
        ]

        self.__acceleration_method = acceleration_method

        super().__init__(_sequence_transformers)

    @property
    def acceleration_method(self) -> AccelerationMethod:
        """The acceleration method."""
        return self.__acceleration_method

    @acceleration_method.setter
    def acceleration_method(self, acceleration_method: AccelerationMethod) -> None:
        self._sequence_transformers[1] = self.__sequence_transformer_factory.create(
            acceleration_method
        )
        self.__acceleration_method = acceleration_method
        self.clear()

    @property
    def over_relaxation_factor(self) -> float:
        """The over-relaxation factor."""
        return self._sequence_transformers[0].factor

    @over_relaxation_factor.setter
    def over_relaxation_factor(self, over_relaxation_factor: float) -> None:
        self._sequence_transformers[0].factor = over_relaxation_factor
        self.clear()
