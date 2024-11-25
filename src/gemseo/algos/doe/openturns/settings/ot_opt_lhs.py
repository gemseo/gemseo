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
"""Settings for the optimized LHS DOE from the OpenTURNS library."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt  # noqa: TC002

from gemseo.algos.doe.openturns._algos.ot_optimal_lhs import OTOptimalLHS
from gemseo.algos.doe.openturns.settings.base_openturns_settings import (
    BaseOpenTURNSSettings,
)

TemperatureProfile = OTOptimalLHS.TemperatureProfile

SpaceFillingCriterion = OTOptimalLHS.SpaceFillingCriterion


class OT_OPT_LHS_Settings(BaseOpenTURNSSettings):  # noqa: N801
    """The settings for the optimized LHS DOE from the OpenTURNS library."""

    _TARGET_CLASS_NAME = "OT_OPT_LHS"

    n_samples: PositiveInt = Field(description="""The number of samples.""", ge=2)

    temperature: TemperatureProfile = Field(
        default=TemperatureProfile.GEOMETRIC,
        description="""The temperature profile for simulated annealing.

Either "Geometric" or "Linear".""",
    )

    criterion: SpaceFillingCriterion = Field(
        default=SpaceFillingCriterion.C2,
        description="The space-filling criterion.",
    )

    annealing: bool = Field(
        default=True,
        description="""Whether to use simulated annealing to optimize the LHS.

If ``False``, the crude Monte Carlo method is used.""",
    )

    n_replicates: PositiveInt = Field(
        default=1_000,
        description="The number of Monte Carlo replicates to optimize LHS.",
    )
