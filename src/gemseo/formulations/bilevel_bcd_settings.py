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
"""Settings of the BiLevel BCD formulation ."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator

from gemseo.formulations.bilevel_settings import BiLevel_Settings
from gemseo.mda.base_mda_settings import BaseMDASettings
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.utils.pydantic import copy_field

if TYPE_CHECKING:
    from typing_extensions import Self


copy_field_opt = partial(copy_field, model=BaseMDASettings)


class BiLevel_BCD_Settings(BiLevel_Settings):  # noqa: N801
    """Settings of the :class:`.BiLevel` formulation."""

    _TARGET_CLASS_NAME = "BiLevelBCD"

    bcd_mda_settings: MDAGaussSeidel_Settings = Field(
        default=MDAGaussSeidel_Settings(warm_start=True),
        description="The settings for the MDA used in the BCD method.",
    )

    @model_validator(mode="after")
    def __force_bcd_mda_warm_start(self) -> Self:
        """Validates the state of the BCD MDA warm start setting as True."""
        if not self.bcd_mda_settings.warm_start:
            self.bcd_mda_settings.warm_start = True
        return self
