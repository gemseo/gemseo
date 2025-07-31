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
"""Base settings for defining a uniform distribution."""

from __future__ import annotations

from typing import Final

from pydantic import Field
from pydantic import PositiveFloat

from gemseo.uncertainty.distributions.base_distribution_settings import (
    BaseDistribution_Settings,
)

_LOCATION: Final[float] = 0.0
"""The default value of location."""

_SCALE: Final[float] = 1.0
"""The default value of scale."""

_SHAPE: Final[float] = 1.0
"""The default value of shape."""

_USE_WEIBULL_MIN: Final[bool] = True
"""The default value of use_weibull_min."""


class BaseWeibullDistribution_Settings(BaseDistribution_Settings):  # noqa: N801
    """The base settings of a uniform distribution."""

    _TARGET_CLASS_NAME = "SPWeibullDistribution"

    location: float = Field(
        default=_LOCATION,
        description=(
            r"The location parameter :math:`\gamma` of the Weibull distribution."
        ),
    )

    scale: PositiveFloat = Field(
        default=_SCALE,
        description="The scale parameter of the Weibull distribution.",
    )

    shape: PositiveFloat = Field(
        default=_SHAPE,
        description="The shape parameter of the Weibull distribution.",
    )

    use_weibull_min: bool = Field(
        default=_USE_WEIBULL_MIN,
        description=r"""Whether to use
the Weibull minimum extreme value distribution
(the support of the random variable is :math:`[\gamma,+\infty[`)
or the Weibull maximum extreme value distribution
(the support of the random variable is :math:`]-\infty[,\gamma]`).""",
    )
