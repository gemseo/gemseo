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
"""The settings for the directional sampling algorithm."""

from __future__ import annotations

from pydantic import Field

from gemseo.uncertainty.reliability.openturns.base_sampling_settings import (
    BaseOTSamplingSettings,
)
from gemseo.uncertainty.reliability.openturns.root_strategy import BaseOTRootStrategy
from gemseo.uncertainty.reliability.openturns.root_strategy import OTSafeAndSlow


class OT_DirectionalSampling_Settings(BaseOTSamplingSettings):  # noqa: N801
    """The settings of the directional sampling algorithm."""

    root_strategy: BaseOTRootStrategy = Field(
        default=OTSafeAndSlow(),
        description="The strategy used to evaluate the frontiers of the event "
        "along each direction in the standard space.",
    )

    use_random_sampling_strategy: bool = Field(
        default=True,
        description="Whether to use the random direction strategy. "
        "Otherwise, use the orthogonal one.",
    )
