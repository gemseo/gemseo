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
"""The alternate 2-δ acceleration method."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import vstack
from scipy.linalg import lstsq

from gemseo.algos.sequence_transformer.sequence_transformer import SequenceTransformer

if TYPE_CHECKING:
    from typing import ClassVar

    from gemseo.typing import NumberArray


class Alternate2Delta(SequenceTransformer):
    """The alternate 2-δ acceleration method.

    The method is introduced in: Isabelle Ramiere, Thomas Helfer, "Iterative residual-
    based vector methods to accelerate fixed point iterations", Computers and
    Mathematics with Applications, (2015) eq. (50).

    The least squares problem that must be solved to perform the transformation may be
    degenerated when the vectors :math:`x_{n+1} - x_n` and :math:`x_n - x_{n-1}` are
    collinear.
    """

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 3
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 3

    def _compute_transformed_iterate(self) -> NumberArray:
        dxn_2, dxn_1, dxn = self._residuals
        gxn_2, gxn_1, gxn = self._iterates

        y, _, rank, _ = lstsq(vstack([dxn - dxn_1, dxn_1 - dxn_2]).T, dxn, cond=1e-10)

        if rank < 2:
            return gxn

        return gxn - y[0] * (gxn - gxn_1) - y[1] * (gxn_1 - gxn_2)
