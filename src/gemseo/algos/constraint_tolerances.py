# Copyright 2022 Airbus SAS
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
"""The equality and inequality constraint tolerances."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import NonNegativeFloat


class ConstraintTolerances(BaseModel):
    """The equality and inequality constraint tolerances."""

    inequality: NonNegativeFloat = 1e-4
    """The inequality constraint tolerances."""

    equality: NonNegativeFloat = 1e-2
    """The equality constraint tolerances."""
