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
"""A factory of tolerance intervals."""

from __future__ import annotations

from gemseo.core.base_factory import BaseFactory
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    BaseToleranceInterval,
)


class ToleranceIntervalFactory(BaseFactory):
    """A factory of tolerance intervals."""

    _CLASS = BaseToleranceInterval
    _MODULE_NAMES = ("gemseo.uncertainty.statistics.tolerance_interval",)

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        for class_info in tuple(self._names_to_class_info.values()):
            class_name = class_info.class_.__name__.replace("ToleranceInterval", "")
            self._names_to_class_info[class_name] = class_info
