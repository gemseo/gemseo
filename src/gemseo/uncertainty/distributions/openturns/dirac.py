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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Reda El Amri
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The OpenTURNS-based Dirac distribution."""

from __future__ import annotations

from gemseo.uncertainty.distributions.openturns.dirac_settings import (
    OTDiracDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.utils.pydantic import create_model


class OTDiracDistribution(OTDistribution):
    """The OpenTURNS-based Dirac distribution."""

    settings_class = OTDiracDistribution_Settings

    def __init__(self, settings: OTDiracDistribution_Settings | None = None) -> None:  # noqa: D107
        settings = create_model(OTDiracDistribution_Settings, settings_model=settings)
        super().__init__(
            OTDistribution_Settings(
                interfaced_distribution="Dirac",
                parameters=(settings.variable_value,),
                standard_parameters={self._LOC: settings.variable_value},
                transformation=settings.transformation,
                lower_bound=settings.lower_bound,
                upper_bound=settings.upper_bound,
                threshold=settings.threshold,
            )
        )
