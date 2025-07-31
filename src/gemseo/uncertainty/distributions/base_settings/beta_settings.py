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
"""Base settings for defining a Beta distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from pydantic import Field
from pydantic import PositiveFloat
from pydantic import model_validator

from gemseo.uncertainty.distributions.base_distribution_settings import (
    BaseDistribution_Settings,
)

if TYPE_CHECKING:
    from typing_extensions import Self

_ALPHA: Final[float] = 2.0
"""The default value of alpha."""

_BETA: Final[float] = 2.0
"""The default value of beta."""

_MAXIMUM: Final[float] = 1.0
"""The default value of maximum."""

_MINIMUM: Final[float] = 0.0
"""The default value of minimum."""


class BaseBetaDistribution_Settings(BaseDistribution_Settings):  # noqa: N801
    """The base settings of a Beta distribution."""

    alpha: PositiveFloat = Field(
        default=_ALPHA,
        description="The first shape parameter of the beta random variable.",
    )

    beta: PositiveFloat = Field(
        default=_BETA,
        description="The second shape parameter of the beta random variable.",
    )

    minimum: float = Field(
        default=_MINIMUM,
        description="The second shape parameter of the beta random variable.",
    )

    maximum: float = Field(
        default=_MAXIMUM,
        description="The maximum of the beta random variable.",
    )

    @model_validator(mode="after")
    def __validate(self) -> Self:
        if self.maximum <= self.minimum:
            msg = (
                "The maximum of the beta random variable must be "
                "greater than its minimum."
            )
            raise ValueError(msg)

        return self
