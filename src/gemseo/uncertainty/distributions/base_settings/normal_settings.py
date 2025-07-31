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
"""Base settings for defining a normal distribution."""

from __future__ import annotations

from typing import Final

from pydantic import Field
from pydantic import PositiveFloat

from gemseo.uncertainty.distributions.base_distribution_settings import (
    BaseDistribution_Settings,
)

_MU: Final[float] = 0.0
"""The default value of mu."""

_SIGMA: Final[float] = 1.0
"""The default value of sigma."""


class BaseNormalDistribution_Settings(BaseDistribution_Settings):  # noqa: N801
    """The base settings of a normal distribution."""

    mu: float = Field(
        default=_MU,
        description="The mean of the normal random variable.",
    )

    sigma: PositiveFloat = Field(
        default=_SIGMA,
        description="The standard deviation of the normal random variable.",
    )
