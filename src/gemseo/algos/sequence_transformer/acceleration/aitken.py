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
"""The Aitken's acceleration method."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.algos.sequence_transformer.sequence_transformer import SequenceTransformer

if TYPE_CHECKING:
    from typing import ClassVar

    from numpy.typing import NDArray


class Aitken(SequenceTransformer):
    """The vector Δ²-Aitken acceleration method.

    The method is introduced in: Isabelle Ramiere, Thomas Helfer, "Iterative residual-
    based vector methods to accelerate fixed point iterations", Computers and
    Mathematics with Applications, (2015) eq. (45).
    """

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 1
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 2

    def _compute_transformed_iterate(self) -> NDArray:
        dxn_1, dxn = self._residuals
        gxn = self._iterates[-1]

        d2xn = dxn - dxn_1

        return gxn - (d2xn.T @ dxn) / (d2xn.T @ d2xn) * dxn
