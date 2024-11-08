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
"""An evaluation counter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationCounter:
    """An evaluation counter."""

    current: int = 0
    """The current number of evaluations."""

    maximum: int = 0
    """The maximum number of evaluations allowed."""

    @property
    def maximum_is_reached(self) -> bool:
        """Whether the maximum number of evaluations is reached."""
        if self.maximum == 0:
            return False

        return self.current >= self.maximum

    def __post_init__(self):
        if self.current > self.maximum:
            msg = (
                f"The current value ({self.current}) of the evaluation counter "
                "must be less than or equal to "
                f"the maximum number of evaluations ({self.maximum})."
            )
            raise ValueError(msg)
