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
"""The settings for the SORM algorithm."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from strenum import StrEnum

from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings


class Approximation(StrEnum):
    """The probability approximation method."""

    BREITUNG = "Breitung"
    HOHENBICHLER = "Hohenbichler"
    TVEDT = "Tvedt"


class OT_SORM_Settings(OT_FORM_Settings):  # noqa: N801
    """The base class for the settings of the SORM algorithm."""

    Approximation: ClassVar[type[Approximation]] = Approximation

    approximation: Approximation = Field(
        default=Approximation.BREITUNG,
        description="The probability approximation method.",
    )
