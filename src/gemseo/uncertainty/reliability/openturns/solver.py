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
"""The OpenTURNS solvers for 1D non-linear equations."""

from __future__ import annotations

from typing import ClassVar

from openturns import Bisection
from openturns import Brent
from openturns import Secant
from openturns import SolverImplementation
from pydantic import BaseModel
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt


class BaseOTSolver(BaseModel):
    """The base class for OpenTURNS solvers for 1D non-linear equations."""

    ALGO_CLASS: ClassVar[type[SolverImplementation]]
    """The type of solver."""

    absolute_error: NonNegativeFloat = Field(
        default=1e-5,
        description="The distance between two successive iterates at the end point.",
    )

    relative_error: NonNegativeFloat = Field(
        default=1e-5,
        description="The distance between the two last successive iterates "
        "with regards to the last iterate.",
    )

    residual_error: NonNegativeFloat = Field(
        default=1e-8,
        description="The difference "
        "between the last iterate value and the expected value.",
    )

    maximum_calls_number: PositiveInt = Field(
        default=100,
        description="The maximum number of evaluations.",
    )


class OTBisection(BaseOTSolver):
    """The bisection algorithm."""

    ALGO_CLASS: ClassVar[type[Bisection]] = Bisection


class OTBrent(BaseOTSolver):
    """The Brent algorithm."""

    ALGO_CLASS: ClassVar[type[Brent]] = Brent


class OTSecant(BaseOTSolver):
    """The secant algorithm."""

    ALGO_CLASS: ClassVar[type[Secant]] = Secant
