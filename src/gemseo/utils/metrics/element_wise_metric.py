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
"""An element-wise metric.

An element-wise metric applies a metric on each element of two collections of the same
size.
"""

from collections.abc import Iterable
from itertools import starmap
from numbers import Number
from typing import Any
from typing import TypeVar

from gemseo.datasets.dataset import Dataset
from gemseo.typing import NumberArray
from gemseo.utils.metrics.base_composite_metric import BaseCompositeMetric

_InputT = TypeVar("_InputT", Iterable[NumberArray], Iterable[Number], Dataset)


class ElementWiseMetric(BaseCompositeMetric[_InputT, Any]):
    """An element-wise metric."""

    def compute(self, a: _InputT, b: _InputT) -> list[Any]:  # noqa: D102
        return list(starmap(self._metric.compute, zip(a, b)))
