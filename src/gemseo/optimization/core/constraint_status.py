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
"""The status of a constraint at a point."""

from __future__ import annotations

from enum import StrEnum
from enum import auto


class ConstraintStatus(StrEnum):
    """The status of a constraint at a point."""

    SATISFIED = auto()
    """The constraint is satisfied within its tolerance."""

    VIOLATED = auto()
    """The constraint is violated, or its value contains NaN."""

    MISSING = auto()
    """The value of the constraint is absent from the point."""
