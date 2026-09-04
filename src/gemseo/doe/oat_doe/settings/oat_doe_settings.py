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
"""Settings of the OAT DOE."""

from __future__ import annotations

from typing import Final

from pydantic import Field
from pydantic.types import PositiveFloat  # noqa: TC002

from gemseo.doe.core.base_doe_settings import BaseDOESettings
from gemseo.util.pydantic_ndarray import NDArrayPydantic  # noqa: TC001

DEFAULT_STEP: Final[float] = 0.05
"""The default relative step of the OAT method."""


class OATDOE_Settings(BaseDOESettings):  # noqa: N801
    """The settings of the OAT DOE."""

    initial_point: NDArrayPydantic = Field(
        description="The initial point of the OAT DOE."
    )

    step: PositiveFloat = Field(
        default=DEFAULT_STEP,
        lt=0.5,
        description="""The relative step of the OAT DOE.

This step is taken in the unit hypercube:
the `i`-th coordinate `u` of the initial point becomes
`u+step` if `u+step<1` and `u-step` otherwise.
It must be smaller than 0.5
so that the perturbed coordinate stays in the open interval `(0,1)`.""",
    )
