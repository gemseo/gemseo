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
"""Acceleration methods."""
from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import hstack
from numpy import vstack
from scipy.linalg import lstsq

from gemseo.algos.sequence_transformer.sequence_transformer import SequenceTransformer

if TYPE_CHECKING:
    from typing import ClassVar
    from numpy.typing import NDArray


class NoAcceleration(SequenceTransformer):
    """A SequenceTransformer which leaves the sequence unchanged."""

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 0
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 0

    def _compute_transformed_iterate(self) -> None:  # pragma: no cover
        pass

    def compute_transformed_iterate(  # noqa: D102
        self, current_iterate: NDArray, next_iterate: NDArray
    ) -> NDArray:
        return next_iterate


class Alternate2Delta(SequenceTransformer):
    """The alternate 2-δ acceleration method.

    The method is introduced in: Isabelle Ramiere, Thomas Helfer, "Iterative residual-
    based vector methods to accelerate fixed point iterations", Computers and
    Mathematics with Applications, (2015) eq. (50).
    """

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 3
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 3

    def _compute_transformed_iterate(self) -> NDArray:
        dxn_2, dxn_1, dxn = self._residuals
        gxn_2, gxn_1, gxn = self._iterates

        y, _, _, _ = lstsq(vstack([dxn - dxn_1, dxn_1 - dxn_2]).T, dxn, cond=1e-16)

        return gxn - y[0] * (gxn - gxn_1) - y[1] * (gxn_1 - gxn_2)


class AlternateDeltaSquared(SequenceTransformer):
    """The alternate δ² acceleration method.

    The method is introduced in: Isabelle Ramiere, Thomas Helfer, "Iterative residual-
    based vector methods to accelerate fixed point iterations", Computers and
    Mathematics with Applications, (2015) eq. (48).
    """

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 3
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 3

    def _compute_transformed_iterate(self) -> NDArray:
        dxn_2, dxn_1, dxn = self._residuals
        gxn_2, gxn_1, gxn = self._iterates

        dz = dxn - 2 * dxn_1 + dxn_2
        gz = gxn - 2 * gxn_1 + gxn_2

        return gxn - (dz.T @ dxn) / (dz.T @ dz) * gz


class Secant(SequenceTransformer):
    """The secant acceleration method.

    The method is introduced in: Isabelle Ramiere, Thomas Helfer, "Iterative residual-
    based vector methods to accelerate fixed point iterations", Computers and
    Mathematics with Applications, (2015) eq. (45).
    """

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int] = 2
    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int] = 2

    def _compute_transformed_iterate(self) -> NDArray:
        dxn_1, dxn = self._residuals
        gxn_1, gxn = self._iterates

        d2xn = dxn - dxn_1
        dgxn = gxn - gxn_1

        return gxn - (d2xn.T @ dxn) / (d2xn.T @ d2xn) * dgxn


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
