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
"""Settings for the OpenTURNS-based probability distributions."""

from __future__ import annotations

from pydantic import Field

from gemseo.uncertainty.distributions.base_distribution_settings import (
    BaseGenericDistributionSettings,
)
from gemseo.uncertainty.distributions.openturns.base_settings import (
    BaseOTMarginalDistributionSettings,
)


class OTDistribution_Settings(  # noqa: N801
    BaseGenericDistributionSettings, BaseOTMarginalDistributionSettings
):
    """The settings of an OpenTURNS-based distribution."""

    interfaced_distribution: str = Field(
        default="Uniform", description="The name of the probability distribution."
    )

    parameters: tuple[float, ...] = Field(
        default_factory=tuple,
        description="The parameters of the probability distribution.",
    )
