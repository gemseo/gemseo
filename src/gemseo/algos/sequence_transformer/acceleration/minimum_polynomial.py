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
"""The minimum polynomial method."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import hstack
from scipy.linalg import lstsq

from gemseo.algos.sequence_transformer.sequence_transformer import SequenceTransformer

if TYPE_CHECKING:
    from typing import ClassVar

    from numpy.typing import NDArray


class MinimumPolynomial(SequenceTransformer):
    """The minimum polynomial extrapolation method.

    The method is introduced in: Cabay, S.; Jackson, L.W, "A polynomial extrapolation
    method for finding limits and antilimits of vector sequences", SIAM Journal on
    Numerical Analysis, (1976).
    """

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 2
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 2

    def __init__(self, window_size: int = 5) -> None:
        """
        Args:
            window_size: The maximum number of iterates to be kept.
        """  # noqa:D205 D212 D415
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError("The window size must be greater than or equal to 1.")

        self.__window_size = window_size

        self.__d2xn_matrix = None
        self.__dgxn_matrix = None

        super().__init__()

    def _compute_transformed_iterate(self) -> NDArray:
        d2xn = (self._residuals[-1] - self._residuals[-2]).reshape(-1, 1)
        dgxn = (self._iterates[-1] - self._iterates[-2]).reshape(-1, 1)

        # If reaching up the window size, then remove the oldest element
        if (
            self.__d2xn_matrix is not None
            and self.__d2xn_matrix.shape[1] == self.__window_size
        ):
            self.__d2xn_matrix = self.__d2xn_matrix[:, 1:]
            self.__dgxn_matrix = self.__dgxn_matrix[:, 1:]

        # Stack the new vectors to both matrices
        self.__d2xn_matrix = (
            hstack([self.__d2xn_matrix, d2xn])
            if self.__d2xn_matrix is not None
            else d2xn
        )

        self.__dgxn_matrix = (
            hstack([self.__dgxn_matrix, dgxn])
            if self.__dgxn_matrix is not None
            else dgxn
        )

        c, _, _, _ = lstsq(self.__d2xn_matrix, self._residuals[-1], cond=1e-16)

        return self._iterates[-1] - self.__dgxn_matrix @ c
