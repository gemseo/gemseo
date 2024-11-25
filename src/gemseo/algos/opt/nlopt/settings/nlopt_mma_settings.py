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
"""Settings for the NLopt MMA algorithm."""

from __future__ import annotations

from pydantic import Field  # noqa: TC002
from pydantic import NonNegativeInt  # noqa: TC002

from gemseo.algos.opt.base_gradient_based_algorithm_settings import (
    BaseGradientBasedAlgorithmSettings,
)
from gemseo.algos.opt.nlopt.settings.base_nlopt_settings import BaseNLoptSettings


class NLOPT_MMA_Settings(BaseNLoptSettings, BaseGradientBasedAlgorithmSettings):  # noqa: N801
    """The settings for the NLopt MMA algorithm."""

    _TARGET_CLASS_NAME = "NLOPT_MMA"

    inner_maxeval: NonNegativeInt = Field(
        default=0,
        description="""The maximum number of inner iterations of the algorithm.

The value 0 means that there is no limit.""",
    )
