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
"""Settings for the OpenTURNS-based finite discrete distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import PositiveFloat
from pydantic import model_validator

from gemseo.uncertainty.distributions.openturns.base_settings import (
    BaseOTMarginalDistributionSettings,
)

if TYPE_CHECKING:
    from typing_extensions import Self


class OTFiniteDiscreteDistribution_Settings(BaseOTMarginalDistributionSettings):  # noqa: N801
    """The settings of an OpenTURNS-based finite discrete distribution."""

    value_to_weight: dict[tuple[float, ...] | float, PositiveFloat] = Field(
        description="The map from the possible values to the weights.",
    )

    @model_validator(mode="after")
    def __validate(self) -> Self:
        """Validate the settings of the OpenTURNS-based finite discrete distribution."""
        self.__dict__["value_to_weight"] = {
            k if isinstance(k, tuple) else (k,): v
            for k, v in self.value_to_weight.items()
        }
        return self
