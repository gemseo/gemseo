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
"""Settings for the OpenTURNS-based Weibull distributions."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveFloat

from gemseo.uncertainty.distributions.base_settings.weibull_settings import (
    BaseWeibullDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    _OTDistribution_Settings_Mixin,
)


class OTWeibullDistribution_Settings(  # noqa: N801
    BaseWeibullDistribution_Settings, _OTDistribution_Settings_Mixin
):
    """The settings of an OpenTURNS-based uniform distribution."""

    _TARGET_CLASS_NAME = "OTWeibullDistribution"

    location: float = Field(
        default=0.0,
        description=(
            r"The location parameter :math:`\gamma` of the Weibull distribution."
        ),
    )

    scale: PositiveFloat = Field(
        default=1.0,
        description="The scale parameter of the Weibull distribution.",
    )

    shape: PositiveFloat = Field(
        default=1.0,
        description="The shape parameter of the Weibull distribution.",
    )

    use_weibull_min: bool = Field(
        default=True,
        description=r"""Whether to use
the Weibull minimum extreme value distribution
(the support of the random variable is :math:`[\gamma,+\infty[`)
or the Weibull maximum extreme value distribution
(the support of the random variable is :math:`]-\infty[,\gamma]`).""",
    )
