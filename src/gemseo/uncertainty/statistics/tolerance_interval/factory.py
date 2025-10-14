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

from typing import TYPE_CHECKING

from gemseo.core.base_factory import BaseFactory
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    BaseToleranceInterval,
)

if TYPE_CHECKING:
    from gemseo.core.base_factory import _ClassInfo


class ToleranceIntervalFactory(BaseFactory[BaseToleranceInterval]):
    """A factory of tolerance intervals."""

    _CLASS = BaseToleranceInterval
    _PACKAGE_NAMES = ("gemseo.uncertainty.statistics.tolerance_interval",)

    @property
    def _names_to_class_info(self) -> dict[str, _ClassInfo[BaseToleranceInterval]]:
        names_to_class_info = super()._names_to_class_info
        for class_info in tuple(names_to_class_info.values()):
            class_name = class_info.class_.__name__.replace("ToleranceInterval", "")
            names_to_class_info[class_name] = class_info
        return names_to_class_info
