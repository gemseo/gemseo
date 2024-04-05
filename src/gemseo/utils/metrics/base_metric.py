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
"""A base metric class to compare two quantities."""

from __future__ import annotations

from abc import abstractmethod
from typing import Generic
from typing import TypeVar

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT")


class BaseMetric(
    Generic[_InputT, _OutputT], metaclass=ABCGoogleDocstringInheritanceMeta
):
    """A base class to implement metrics.

    A metric is used to compare two quantities ``a`` and ``b``.
    """

    @abstractmethod
    def compute(self, a: _InputT, b: _InputT) -> _OutputT:
        """Evaluate the metric.

        Args:
            a: A first quantity.
            b: A second quantity.

        Returns:
            The metric associated with ``a`` and ``b``.
        """
