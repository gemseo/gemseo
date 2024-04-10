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

from typing import Any

from gemseo.core.base_factory import BaseFactory
from gemseo.uncertainty.statistics.tolerance_interval.distribution import LOGGER
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    BaseToleranceInterval,
)


class ToleranceIntervalFactory(BaseFactory):
    """A factory of :class:`.BaseToleranceInterval`."""

    _CLASS = BaseToleranceInterval
    _MODULE_NAMES = ("gemseo.uncertainty.statistics.tolerance_interval",)

    def create(
        self,
        class_name: str,
        size: int,
        *args: float,
    ) -> BaseToleranceInterval:
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
            LOGGER.exception(
                "Failed to create class %s with arguments %s", class_name, args
            )
            msg = f"Cannot create {class_name}ToleranceInterval with arguments {args}"
            raise RuntimeError(msg) from None

    def get_class(self, name: str) -> type[Any]:
        """Return a class from its name.

        Args:
            name: The name of the class.

        Returns:
            The class.
        """
        return super().get_class(f"{name}ToleranceInterval")
