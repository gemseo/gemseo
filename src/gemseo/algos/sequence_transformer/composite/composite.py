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
"""A composite of sequence transformers applied sequentially."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.sequence_transformer.sequence_transformer import SequenceTransformer

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import ClassVar

    from numpy.typing import NDArray


class CompositeSequenceTransformer(SequenceTransformer):
    """A composite of SequenceTransformer."""

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 0
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 0

    _sequence_transformers: Iterable[SequenceTransformer]
    """The sequence transformers that are chained."""

    def __init__(self, sequence_transformers: Iterable[SequenceTransformer]) -> None:
        """
        Args:
            sequence_transformers: The sequence of SequenceTransformers.
        """  # noqa:D205 D212 D415
        super().__init__()

        self._sequence_transformers = sequence_transformers

    def _compute_transformed_iterate(self) -> None:  # pragma: no cover
        pass

    def clear(self) -> None:
        """Clear the iterates in the double-ended queues."""
        for transformer in self._sequence_transformers:
            transformer.clear()

    def compute_transformed_iterate(
        self,
        iterate: NDArray,
        residual: NDArray,
    ) -> NDArray:
        """Compute the next transformed iterate.

        Args:
            iterate: The iterate :math:`G(x_n)`.
            residual: The associated residual :math:`G(x_n) - x_n`.

        Returns:
            The next transformed iterate :math:`x_{n+1}`.
        """
        current_iterate = (iterate - residual).copy()
        next_iterate = iterate.copy()

        for transformer in self._sequence_transformers:
            next_iterate = transformer.compute_transformed_iterate(
                next_iterate, next_iterate - current_iterate
            )

        return next_iterate
