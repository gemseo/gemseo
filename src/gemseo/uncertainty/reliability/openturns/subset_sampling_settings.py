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
"""The settings for the subset sampling algorithm."""

from __future__ import annotations

from openturns.SpecFunc import MinScalar
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveFloat

from gemseo.uncertainty.reliability.openturns.base_sampling_settings import (
    BaseOTSamplingSettings,
)


class OT_SubsetSampling_Settings(BaseOTSamplingSettings):  # noqa: N801
    """The settings of the subset sampling algorithm."""

    minimum_probability: NonNegativeFloat = Field(
        default=MinScalar,
        description="Allows one to stop the algorithm "
        "if the probability becomes too small.",
    )

    proposal_range: NonNegativeFloat = Field(
        default=0.0, description="The proposal range length."
    )

    target_probability: PositiveFloat = Field(
        default=0.5,
        description=r"The conditional failure probability $\mathbb{P}[F_i|F_{i-1}]$ "
        "in the expression "
        r"$\mathbb{P}[F]=\mathbb{P}[F_1]\prod_{i=2}^m\mathbb{P}[F_i|F_{i-1}]$.",
        lt=1.0,
    )
