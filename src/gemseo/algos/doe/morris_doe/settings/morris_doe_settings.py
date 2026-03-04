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

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import model_validator

from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.algos.doe.pydoe.settings.pydoe_lhs import PYDOE_LHS_Settings

if TYPE_CHECKING:
    from typing_extensions import Self


class MorrisDOE_Settings(BaseDOESettings):  # noqa: N801
    """The `MorrisDOE` settings."""

    n_samples: NonNegativeInt = Field(
        default=0,
        description="The maximum number of samples required by the user. "
        "If 0, "
        "the effective number of samples is equal to the design space dimension "
        "multiplied by one plus the number of initial points "
        "defined by the `n_samples` field of `doe_algo_settings`."
        "Otherwise, "
        "the `n_samples` field of `doe_algo_settings` will be inferred from it. ",
    )

    doe_algo_settings: BaseDOESettings = Field(
        default_factory=PYDOE_LHS_Settings,
        description="The settings of the DOE algorithm to generate the initial points. "
        "Its `n_samples` field, if any, is ignored when the main `n_samples` is not 0.",
    )

    step: PositiveFloat = Field(
        default=0.05,
        description="The relative step of the OAT DOE.",
    )

    @model_validator(mode="after")
    def __validate(self) -> Self:
        """Validate the settings."""
        if (
            self.n_samples > 0
            and "n_samples" not in self.doe_algo_settings.model_fields
        ):
            msg = "When n_samples > 0, doe_algo_settings must have an n_samples field."
            raise ValueError(msg)
        return self
