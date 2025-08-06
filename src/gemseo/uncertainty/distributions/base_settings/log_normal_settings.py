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
"""Base settings for defining a log-normal distribution."""

from __future__ import annotations

from typing import Final

from pydantic import Field
from pydantic import PositiveFloat

from gemseo.uncertainty.distributions.base_distribution_settings import (
    BaseDistribution_Settings,
)

_MU: Final[float] = 1.0
"""The default value of mu."""

_SIGMA: Final[float] = 1.0
"""The default value of sigma."""

_LOCATION: Final[float] = 0.0
"""The default value of location."""

_SET_LOG: Final[bool] = False
"""The default value of set_log."""


class BaseLogNormalDistribution_Settings(BaseDistribution_Settings):  # noqa: N801
    """The base settings of a log-normal distribution."""

    mu: float = Field(
        default=_MU,
        description="""Either the mean of the log-normal random variable
or that of its logarithm when ``set_log`` is ``True``.""",
    )

    sigma: PositiveFloat = Field(
        default=_SIGMA,
        description="""Either the standard deviation of the log-normal random variable
or that of its logarithm when ``set_log`` is ``True``.""",
    )

    location: float = Field(
        default=_LOCATION,
        description="The location of the log-normal random variable.",
    )

    set_log: bool = Field(
        default=_SET_LOG,
        description="""Whether ``mu`` and ``sigma`` apply
to the logarithm of the log-normal random variable.
Otherwise,
``mu`` and ``sigma`` apply to the log-normal random variable directly.""",
    )
