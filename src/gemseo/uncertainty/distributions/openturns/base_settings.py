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
"""Settings for OpenTURNS-based probability distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from pydantic import Field
from pydantic import model_validator

from gemseo.uncertainty.distributions.base_settings import BaseDistributionSettings

if TYPE_CHECKING:
    from typing_extensions import Self


class BaseOTDistributionSettings(BaseDistributionSettings):  # noqa: N801
    """The base class for the settings of an OpenTURNS-based probability distribution."""  # noqa: E501

    _LIBRARY_NAME: ClassVar[str] = "OpenTURNS"


class BaseOTMarginalDistributionSettings(BaseOTDistributionSettings):  # noqa: N801
    """The base class for the settings of an OpenTURNS-based marginal probability distribution."""  # noqa: E501

    transformation: str = Field(
        default="",
        description=r"""A transformation applied
to the random variable, e.g. $\sin(x)$. If empty, no transformation.""",
    )

    lower_bound: float | None = Field(
        default=None,
        description="""A lower bound to truncate the probability distribution.

If `None`, no lower truncation.""",
    )

    upper_bound: float | None = Field(
        default=None,
        description="""An upper bound to truncate the probability distribution.

If `None`, no upper truncation.""",
    )

    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=r"""A truncation threshold in $[0,1]$
([see OpenTURNS documentation](http://openturns.github.io/openturns/latest/user_manual/_generated/openturns.TruncatedDistribution.html)).""",
    )

    @model_validator(mode="after")
    def __validate(self) -> Self:
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.upper_bound <= self.lower_bound
        ):
            msg = (
                "The upper truncation bound of a probability distribution must be "
                "greater than its lower truncation bound."
            )
            raise ValueError(msg)

        return self
