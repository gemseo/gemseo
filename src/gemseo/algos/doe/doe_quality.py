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
"""DOE assessor."""
from __future__ import annotations

import logging
from collections import namedtuple
from collections.abc import Callable
from numbers import Real
from operator import ge
from operator import gt
from operator import le
from operator import lt
from typing import Any

from numpy import ndarray
from scipy.spatial import distance
from scipy.stats import qmc

from gemseo.utils.python_compatibility import Final
from gemseo.utils.python_compatibility import Literal

LOGGER = logging.getLogger(__name__)

__EUCLIDEAN: Final[str] = "euclidean"
_DEFAULT_DISCREPANCY_TYPE_NAME: Final[str] = "CD"
_DEFAULT_POWER: Final[int] = 50


DOEMeasures = namedtuple("DOEMeasures", ["discrepancy", "mindist", "phip"])
r"""The quality measures of a DOE.

Namely :math:`\phi^p`, minimum-distance and discrepancy measures,
accessible with the attributes ``discrepancy``, ``mindist`` and ``phip``.

The smaller the quality measures, the better,
except for the minimum-distance criterion for which the larger it is the better.
"""

_measure_transformations: tuple[Callable[[float], float]] = (
    lambda x: x,
    lambda x: -x,
    lambda x: x,
)
"""Transformations of quality measures into quantities to minimize."""

DiscrepancyTypeNameType = Literal["CD", "WD", "MD", "L2-star"]


class DOEQuality:
    """The quality of a DOE."""

    measures: DOEMeasures
    """The quality measures of the DOE."""

    def __init__(
        self,
        samples: ndarray,
        power: int = _DEFAULT_POWER,
        discrepancy_type_name: DiscrepancyTypeNameType = _DEFAULT_DISCREPANCY_TYPE_NAME,
        **discrepancy_options: Any,
    ) -> None:
        r"""
        Args:
            samples: The input samples of the DOE.
            power: The power :math:`p` of the :math:`\phi^p` criterion.
            discrepancy_type_name: The type of discrepancy.
            **discrepancy_options: The options
                passed to ``scipy.stats.qmc.discrepancy``.
        """  # noqa: D205, D212, D415
        self.measures = DOEMeasures(
            compute_discrepancy(
                samples,
                type_name=discrepancy_type_name,
                **discrepancy_options,
            ),
            compute_mindist_criterion(samples),
            compute_phip_criterion(samples, power),
        )

    def __repr__(self):
        return repr(self.measures)

    def __eq__(self, other_doe_quality: DOEQuality) -> bool:
        return all(x == y for x, y in zip(self.measures, other_doe_quality.measures))

    def __lt__(self, other_doe_quality: DOEQuality) -> bool:
        return self.__compare(gt, other_doe_quality)

    def __le__(self, other_doe_quality: DOEQuality) -> bool:
        return self.__compare(ge, other_doe_quality)

    def __gt__(self, other_doe_quality: DOEQuality) -> bool:
        return self.__compare(lt, other_doe_quality)

    def __ge__(self, other_doe_quality: DOEQuality) -> bool:
        return self.__compare(le, other_doe_quality)

    def __compare(
        self, operator: Callable[[Real, Real], bool], other_doe_quality: DOEQuality
    ) -> bool:
        """Compare a DOE quality with another one.

        Args:
            operator: The logical operator.
            other_doe_quality: The other DOE quality.

        Returns:
            Whether the comparison is true.
        """
        return (
            sum(
                operator(transformation(field), transformation(other_field))
                for field, other_field, transformation in zip(
                    self.measures,
                    other_doe_quality.measures,
                    _measure_transformations,
                )
            )
            / len(self.measures)
            >= 0.5
        )


def compute_mindist_criterion(samples: ndarray) -> float:
    """Compute the minimum-distance criterion of a sample set (the higher, the better).

    This criterion is also called *mindist*.

    Args:
        samples: The data samples.

    Returns:
        The minimum-distance criterion.
    """
    return min(distance.pdist(samples, __EUCLIDEAN))


def compute_discrepancy(
    samples: ndarray,
    type_name: DiscrepancyTypeNameType = _DEFAULT_DISCREPANCY_TYPE_NAME,
    **options: Any,
) -> float:
    """Compute the discrepancy of a sample set (the smaller, the better).

    Args:
        samples: The data samples.
        type_name: The type of discrepancy.
        **options: The options passed to :func:`scipy.stats.qmc.discrepancy`.

    Returns:
        The discrepancy.
    """
    return qmc.discrepancy(samples, method=type_name, **options)


def compute_phip_criterion(samples: ndarray, power: float = _DEFAULT_POWER) -> float:
    r"""Compute the math:`\phi^p` criterion of a sample set (the smaller, the better).

    See :cite:`morris1995`.

    Args:
        samples: The data samples.
        power: The power :math:`p` of the :math:`\phi^p` criterion.

    Returns:
        The math:`\phi^p` criterion.
    """
    return sum(distance.pdist(samples, __EUCLIDEAN) ** (-power)) ** (1.0 / power)
