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
"""A squared-error metric."""

from __future__ import annotations

from typing import TypeVar
from typing import Union

from gemseo.typing import NumberArray
from gemseo.utils.metrics.base_metric import BaseMetric

_InputT = TypeVar("_InputT", Union[float, int, complex], NumberArray)


class SquaredErrorMetric(BaseMetric[_InputT, _InputT]):
    """A squared error metric."""

    @staticmethod
    def compute(a: _InputT, b: _InputT) -> _InputT:  # noqa: D102
        return (a - b) ** 2
