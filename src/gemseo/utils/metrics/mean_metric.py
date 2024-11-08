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
"""A mean metric.

A mean metric applies the mean operator on the result
returned by :meth:`.ElementWiseMetric.compute`.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Number
from typing import Any
from typing import TypeVar

from numpy import nanmean

from gemseo.datasets.dataset import Dataset
from gemseo.typing import NumberArray
from gemseo.utils.metrics.base_composite_metric import BaseCompositeMetric

_InputT = TypeVar("_InputT", Dataset, Iterable[Number])


class MeanMetric(BaseCompositeMetric[_InputT, NumberArray]):
    """The mean of an element-wise metric."""

    def __init__(self, metric: BaseCompositeMetric[Any, Any]):
        """
        Args:
            metric: The metric applied at element level.
        """  # noqa: D205, D212, D415
        self._metric = metric

    def compute(  # noqa: D102
        self, a: _InputT, b: _InputT
    ) -> NumberArray:
        return nanmean(self._metric.compute(a, b))  # type: ignore[no-any-return]
