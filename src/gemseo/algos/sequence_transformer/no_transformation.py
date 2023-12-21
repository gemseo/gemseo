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
"""The sequence transformer which does nothing."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.algos.sequence_transformer.sequence_transformer import SequenceTransformer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NoTransformation(SequenceTransformer):
    """A SequenceTransformer which leaves the sequence unchanged."""

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 0
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 0

    def _compute_transformed_iterate(self) -> None:  # pragma: no cover
        pass

    def compute_transformed_iterate(  # noqa: D102
        self, iterate: NDArray, residual: NDArray
    ) -> NDArray:
        return iterate
