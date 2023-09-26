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
#        :author: Charlie Vanaret, Benoit Pauwels, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Sequence transformer methods."""
from __future__ import annotations

from abc import abstractmethod
from collections import deque
from typing import TYPE_CHECKING

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import ClassVar


class SequenceTransformer(metaclass=ABCGoogleDocstringInheritanceMeta):
    r"""A vector sequence transformer for fixed-point iteration method.

    For any function :math:`G : \mathbb{R}^n \rightarrow \mathbb{R}^n`, the fixed point
    iteration method computes the sequence :math:`x_{n+1} = G(x_n)`, whih is exepcted to
    converge towards a fixed point of :math:`G`.

    A sequence transformer is a function :math:`f : \mathbb{R}^n \rightarrow
    \mathbb{R}^n` so that the new iterate is instead computed as
    .. math::
        x_{n+1}' = f(G, x_n, \dots, x_{n-k}, G(x_n)),

    for a given :math:`k \geq 0`.

    The transformed sequence is expected to exhibit faster convergence and/or better
    numerical stability.
    """

    _MINIMUM_NUMBER_OF_ITERATES: ClassVar[int]
    """The minimum number of :math:`G(x_i)` required to compute the transformation."""

    _MINIMUM_NUMBER_OF_RESIDUALS: ClassVar[int]
    """The minimum number of residuals :math:`G(x_i) - x_i` required to compute the
    transformation."""

    _iterates: deque
    """The previously computed iterates :math:`G(x_i)`."""

    _residuals: deque
    """The previously computed residuals :math:`G(x_i) - x_i`."""

    def __init__(self) -> None:  # noqa:D107
        # Instantiate double-ended queues to store the relevant quantities
        self._iterates = deque(maxlen=self._MINIMUM_NUMBER_OF_ITERATES)
        self._residuals = deque(maxlen=self._MINIMUM_NUMBER_OF_RESIDUALS)

    def clear(self) -> None:
        """Clear the iterates."""
        self._iterates.clear()
        self._residuals.clear()

    def compute_transformed_iterate(
        self,
        current_iterate: NDArray,
        next_iterate: NDArray,
    ) -> NDArray:
        """Compute the next transformed iterate.

        Args:
            current_iterate: The current iterate :math:`x_n`.
            next_iterate: The new iterate :math:`G(x_n)`.

        Returns:
            The next transformed iterate :math:`x_{n+1}`.
        """
        # Store iterates and residuals
        self._iterates.append(next_iterate.copy())
        self._residuals.append(next_iterate - current_iterate)

        # Compute the transformed iterate only if sufficient material at hand
        if (
            len(self._iterates) >= self._MINIMUM_NUMBER_OF_ITERATES
            and len(self._residuals) >= self._MINIMUM_NUMBER_OF_RESIDUALS
        ):
            return self._compute_transformed_iterate()
        else:
            return next_iterate

    @abstractmethod
    def _compute_transformed_iterate(self) -> NDArray:
        """Compute the next transformed iterate."""
