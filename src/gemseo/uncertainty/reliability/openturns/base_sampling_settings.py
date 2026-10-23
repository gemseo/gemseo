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
"""The settings for the sampling algorithms."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt
from pydantic import PositiveInt

from gemseo.uncertainty.reliability.core.base_settings import (
    BaseReliabilityAlgorithmSettings,
)


class BaseOTSamplingSettings(BaseReliabilityAlgorithmSettings):  # noqa: N801
    """The base class for the settings of the sampling algorithm."""

    maximum_coefficient_of_variation: NonNegativeFloat = Field(
        default=0.1,
        description="The maximum coefficient of variation of the simulated sample.",
    )

    maximum_outer_sampling: PositiveInt = Field(
        default=1000,
        description="The maximum number of iterations, "
        "each iteration performing a block of evaluations.",
    )

    maximum_standard_deviation: NonNegativeFloat = Field(
        default=0.0, description="The maximum standard deviation of the estimator."
    )

    seed: NonNegativeInt = Field(
        default=0, description="The seed for reproducible results."
    )
