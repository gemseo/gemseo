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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Computation of tolerance intervals from a data-fitted probability distribution."""
from __future__ import annotations

import logging
from typing import Any

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import array
from numpy import inf
from numpy import ndarray

from gemseo.core.factory import Factory
from gemseo.utils.base_enum import BaseEnum

LOGGER = logging.getLogger(__name__)

ToleranceIntervalSide = BaseEnum("ToleranceIntervalSide", "LOWER UPPER BOTH")


class ToleranceInterval(metaclass=GoogleDocstringInheritanceMeta):
    """Computation of tolerance intervals from a data-fitted probability distribution.

    A :class:`.ToleranceInterval` (TI) is initialized
    from the number of samples
    used to estimate the parameters of the probability distribution
    and from the estimations of these parameters.

    A :class:`.ToleranceInterval` can be evaluated from:

    - a coverage defining the minimum percentage of belonging to the TI, e.g. 0.90,
    - a level of confidence in [0,1], e.g. 0.95,
    - a type of interval,
      either 'lower' for lower-sided TI,
      'upper' for upper-sided TI
      or 'both' for both-sided TI.

    .. note::

       Lower-sided tolerance intervals are used to analyse the strength of materials.
       They are also known as *basis tolerance limits*.
       In particular,
       the *B-value* is the lower bound of the lower-sided tolerance interval
       with 90%-coverage and 95%-confidence
       while the *A-value* is the lower bound of the lower-sided tolerance interval
       with 95%-coverage and 95%-confidence.
    """

    def __init__(
        self,
        size: int,
    ) -> None:
        """
        Args:
            size: The number of samples.
        """  # noqa: D205 D212 D415
        self.__size = size

    def _compute_lower_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        """Compute the lower bound of the tolerance interval.

        Args:
            coverage: A minimum percentage of belonging to the TI.
            alpha: ``1-alpha`` is the level of confidence in [0,1].
            size: The number of samples.

        Returns:
            The lower bound of the tolerance interval.
        """
        raise NotImplementedError

    def _compute_upper_bound(
        self,
        coverage: float,
        alpha: float,
        size: int,
    ) -> float:
        """Compute the upper bound of the tolerance interval.

        Args:
            coverage: A minimum percentage of belonging to the TI.
            alpha: ``1-alpha`` is the level of confidence in [0,1].
            size: The number of samples.

        Returns:
            The upper bound of the tolerance interval.
        """
        raise NotImplementedError

    def _compute(
        self,
        coverage: float,
        alpha: float,
        size: int,
        side: ToleranceIntervalSide,
    ) -> tuple[ndarray, ndarray]:
        r"""Compute the bounds of the tolerance interval.

        Args:
            coverage: A minimum percentage of belonging to the TI.
            alpha: ``1-alpha`` is the level of confidence in [0,1].
            size: The number of samples.
            side: The type of the tolerance interval
                characterized by its *sides* of interest,
                either a lower-sided tolerance interval :math:`[a, +\infty[`,
                an upper-sided tolerance interval :math:`]-\infty, b]`,
                or a two-sided tolerance interval :math:`[c, d]`.

        Returns:
            The bounds of the tolerance interval.

        Raises:
            ValueError: If the type of tolerance interval is incorrect.
        """
        if side == ToleranceIntervalSide.LOWER:
            lower = self._compute_lower_bound(coverage, alpha, size)
            upper = inf
        elif side == ToleranceIntervalSide.UPPER:
            lower = -inf
            upper = self._compute_upper_bound(coverage, alpha, size)
        elif side == ToleranceIntervalSide.BOTH:
            coverage = (coverage + 1.0) / 2.0
            alpha /= 2.0
            lower = self._compute_lower_bound(coverage, alpha, size)
            upper = self._compute_upper_bound(coverage, alpha, size)
        else:
            raise ValueError("The type of tolerance interval is incorrect.")
        return array([lower]), array([upper])

    def compute(
        self,
        coverage: float,
        confidence: float = 0.95,
        side: ToleranceIntervalSide = ToleranceIntervalSide.BOTH,
    ) -> tuple[ndarray, ndarray]:
        r"""Compute a tolerance interval.

        Args:
            coverage: A minimum percentage of belonging to the TI.
            confidence: A level of confidence in [0,1].
            side: The type of the tolerance interval
                characterized by its *sides* of interest,
                either a lower-sided tolerance interval :math:`[a, +\infty[`,
                an upper-sided tolerance interval :math:`]-\infty, b]`,
                or a two-sided tolerance interval :math:`[c, d]`.

        Returns:
            The tolerance bounds.
        """
        return self._compute(coverage, 1 - confidence, self.__size, side)


class ToleranceIntervalFactory:
    """A factory of :class:`.ToleranceInterval`."""

    def __init__(self) -> None:  # noqa: D107
        self.__factory = Factory(
            ToleranceInterval, ("gemseo.uncertainty.statistics.tolerance_interval",)
        )

    def create(
        self,
        class_name: str,
        size: int,
        *args: float,
    ) -> ToleranceInterval:
        """Return an instance of :class:`.ToleranceInterval`.

        Args:
            size: The number of samples
                used to estimate the parameters of the probability distribution.
            *args: The arguments of the probability distribution.

        Returns:
            The instance of the class.

        Raises:
            TypeError: If the class cannot be instantiated.
        """
        cls = self.get_class(class_name)
        try:
            return cls(size, *args)
        except TypeError:
            LOGGER.error(
                "Failed to create class %s with arguments %s", class_name, args
            )
            raise TypeError(
                "Cannot create {}ToleranceInterval with arguments {}".format(
                    class_name, args
                )
            )

    def get_class(self, name: str) -> type[Any]:
        """Return a class from its name.

        Args:
            name: The name of the class.

        Returns:
            The class.
        """
        return self.__factory.get_class(f"{name}ToleranceInterval")
