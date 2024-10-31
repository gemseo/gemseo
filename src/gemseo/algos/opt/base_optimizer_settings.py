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
"""Settings for the optimization algorithms."""

from __future__ import annotations

from typing import Any
from typing import ClassVar

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt
from pydantic import model_validator

from gemseo.algos.base_driver_settings import BaseDriverSettings


class BaseOptimizerSettings(BaseDriverSettings):
    """The common parameters for all optimization libraries."""

    ftol_rel: NonNegativeFloat = Field(
        default=0.0,
        description="""The relative tolerance on the objective function.""",
    )

    ftol_abs: NonNegativeFloat = Field(
        default=0.0,
        description="""The absolute tolerance on the objective function.""",
    )

    max_iter: PositiveInt = Field(
        default=1_000,
        description="""The maximum number of iterations.""",
    )

    scaling_threshold: NonNegativeFloat | None = Field(
        default=None,
        description="""The threshold on the reference function value that triggers scaling.

If ``None``, do not scale the functions.""",  # noqa: E501
    )

    stop_crit_n_x: int = Field(
        default=3,
        ge=2,
        description=(
            "The minimum number of design vectors to consider in the stopping criteria."
        ),
    )

    xtol_rel: NonNegativeFloat = Field(
        default=0.0,
        description="""The relative tolerance on the design parameters.""",
    )

    xtol_abs: NonNegativeFloat = Field(
        default=0.0,
        description="""The absolute tolerance on the design parameters.""",
    )

    _redundant_settings: ClassVar[list[str]] = []
    """The settings that has a GEMSEO counterpart.

If such a setting is passed to the library by the end-user, it will be removed.
The user should rather use the corresponding setting from GEMSEO."""

    @model_validator(mode="before")
    @classmethod
    def remove_redundant_settings(cls, data: Any) -> Any:  # noqa: D102
        for setting_name in cls._redundant_settings:
            if setting_name in data:
                msg = (
                    f"The '{setting_name}' setting cannot be passed to the "
                    "optimization library since there exists a GEMSEO counterpart. \n"
                    "Please consider using the corresponding GEMSEO setting (see "
                    "https://gemseo.readthedocs.io/en/stable/algorithms/index.html"
                    "for more informations."
                )
                raise ValueError(msg)

        return data
