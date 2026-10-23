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
"""The settings for the FORM algorithm."""

from __future__ import annotations

from pydantic import Field

from gemseo.uncertainty.reliability.core.base_settings import (
    BaseReliabilityAlgorithmSettings,
)
from gemseo.uncertainty.reliability.openturns.optimizer import BaseOTOptimizer
from gemseo.uncertainty.reliability.openturns.optimizer import OTCobyla


class OT_FORM_Settings(BaseReliabilityAlgorithmSettings):  # noqa: N801
    """The base class for the settings of the FORM algorithm."""

    optimizer: BaseOTOptimizer = Field(
        default_factory=OTCobyla,
        description="The settings of an OpenTURNS optimization algorithm.",
    )
