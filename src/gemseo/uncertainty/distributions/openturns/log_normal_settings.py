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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Settings for the OpenTURNS-based log-normal distributions."""

from __future__ import annotations

from gemseo.uncertainty.distributions.base_settings.log_normal_settings import (
    BaseLogNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    _OTDistribution_Settings_Mixin,
)


class OTLogNormalDistribution_Settings(  # noqa: N801
    BaseLogNormalDistribution_Settings, _OTDistribution_Settings_Mixin
):
    """The settings of an OpenTURNS-based log-normal distribution."""

    _TARGET_CLASS_NAME = "OTLogNormalDistribution"
