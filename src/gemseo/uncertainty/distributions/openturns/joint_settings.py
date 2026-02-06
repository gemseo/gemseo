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
"""Settings for the OpenTURNS-based joint probability distributions."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003

from openturns import DistributionImplementation  # noqa: TC002
from pydantic import Field

from gemseo.uncertainty.distributions.base_distribution_settings import (  # noqa: TC001
    BaseDistributionSettings,
)
from gemseo.uncertainty.distributions.base_joint_settings import (
    BaseJointDistributionSettings,
)
from gemseo.uncertainty.distributions.openturns.base_settings import (
    BaseOTDistributionSettings,
)


class OTJointDistribution_Settings(  # noqa: N801
    BaseJointDistributionSettings, BaseOTDistributionSettings
):
    """The settings of a OpenTURNS-based joint probability distribution."""

    marginal_settings: Sequence[BaseDistributionSettings] = Field(
        description="The OpenTURNS-based marginal probability distributions."
    )

    copula: DistributionImplementation | None = Field(
        default=None,
        description="A copula distribution"
        "defining the dependency structure between random variables;"
        "if `None`, consider an independent copula.",
    )
