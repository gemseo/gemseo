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
"""A metric that relies on another metric."""

from __future__ import annotations

from typing import Any

from gemseo.utils.metrics.base_metric import BaseMetric
from gemseo.utils.metrics.base_metric import _InputT
from gemseo.utils.metrics.base_metric import _OutputT


class BaseCompositeMetric(BaseMetric[_InputT, _OutputT]):
    """A base class for a metric that relies on another metric."""

    _metric: BaseMetric[Any, Any]
    """The metric the composite metric relies on."""

    def __init__(self, metric: BaseMetric[Any, Any]):
        """
        Args:
            metric: The metric the composite metric relies on.
        """  # noqa: D205, D212, D415
        self._metric = metric
