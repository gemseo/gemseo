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
"""Settings for the augmented Lagrangian with penalty update algorithm."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveFloat  # noqa: TC002

from gemseo.algos.opt.augmented_lagrangian.settings.base_augmented_lagrangian_settings import (  # noqa: E501
    BaseAugmentedLagragianSettings,
)


class PenaltyHeuristicSettings(BaseAugmentedLagragianSettings):
    """The augmented Lagrangian with penalty update settings."""

    tau: PositiveFloat = Field(
        default=0.9,
        description="""The threshold to increase the penalty.""",
    )

    gamma: PositiveFloat = Field(
        default=1.5,
        description="""The penalty increase factor.""",
    )

    max_rho: PositiveFloat = Field(
        default=10_000,
        description="""The maximum penalty value.""",
    )
