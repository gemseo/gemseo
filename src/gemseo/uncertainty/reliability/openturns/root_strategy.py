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
"""The OpenTURNS root strategies."""

from __future__ import annotations

from typing import ClassVar

from openturns import MediumSafe
from openturns import RiskyAndFast
from openturns import RootStrategyImplementation
from openturns import SafeAndSlow
from pydantic import BaseModel
from pydantic import Field
from pydantic import PositiveFloat

from gemseo.uncertainty.reliability.openturns.solver import BaseOTSolver
from gemseo.uncertainty.reliability.openturns.solver import OTSecant


class BaseOTRootStrategy(BaseModel):
    """The base class for OpenTURNS root strategies."""

    ALGO_CLASS: ClassVar[type[RootStrategyImplementation]]
    """The OpenTURNS algorithm type."""

    solver: BaseOTSolver = Field(
        default_factory=OTSecant,
        description="The settings of an OpenTURNS solvers for 1D non-linear equations.",
    )

    maximum_distance: PositiveFloat = Field(
        default=8.0,
        description="The distance from the center of the standard space "
        "until which we research an intersection "
        "with the limit state function along each direction.",
    )

    step_size: PositiveFloat = Field(
        default=1.0,
        description="The length of each segment "
        "inside which the root research is performed.",
    )


class OTSafeAndSlow(BaseOTRootStrategy):
    """The SafeAndSlow root strategy."""

    ALGO_CLASS: ClassVar[type[SafeAndSlow]] = SafeAndSlow


class OTRiskyAndFast(BaseOTRootStrategy):
    """The RiskyAndFast root strategy."""

    ALGO_CLASS: ClassVar[type[RiskyAndFast]] = RiskyAndFast


class OTMediumSafe(BaseOTRootStrategy):
    """The MediumSafe root strategy."""

    ALGO_CLASS: ClassVar[type[MediumSafe]] = MediumSafe
