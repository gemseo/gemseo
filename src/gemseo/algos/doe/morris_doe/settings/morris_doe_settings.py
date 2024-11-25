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
"""The settings of the Morris DOE."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat

from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.typing import StrKeyMapping  # noqa: TC001


class MorrisDOE_Settings(BaseDOESettings):  # noqa: N801
    """The ``MorrisDOE`` settings."""

    _TARGET_CLASS_NAME = "MorrisDOE"

    n_samples: NonNegativeInt = Field(
        default=0,
        description=(
            """The maximum number of samples required by the user.

            If 0, deduce it from the design space dimension and ``n_replicates``.
            """
        ),
    )

    doe_algo_name: str = Field(
        default="PYDOE_LHS",
        description="""The name of the DOE algorithm to repeat the OAT DOE.""",
    )

    doe_algo_settings: StrKeyMapping = Field(
        default_factory=dict,
        description="""The options of the DOE algorithm.""",
    )

    step: PositiveFloat = Field(
        default=0.05,
        description="""The relative step of the OAT DOE.""",
    )
