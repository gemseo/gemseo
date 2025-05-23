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
"""Settings for MDAGSNewton."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator

from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings  # noqa: TC001
from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings  # noqa: TC001
from gemseo.mda.sequential_mda_settings import MDASequential_Settings
from gemseo.typing import StrKeyMapping  # noqa: TC001

if TYPE_CHECKING:
    from typing_extensions import Self


class MDAGSNewton_Settings(MDASequential_Settings):  # noqa: N801
    """The settings for :class:`.MDAGSNewton`."""

    _TARGET_CLASS_NAME = "MDAGSNewton"

    gauss_seidel_settings: StrKeyMapping | MDAGaussSeidel_Settings = Field(
        default_factory=dict, description="The settings for the Gauss-Seidel MDA."
    )

    newton_settings: StrKeyMapping | MDANewtonRaphson_Settings = Field(
        default_factory=dict, description="The settings for the Newton-Raphson MDA."
    )

    @model_validator(mode="after")
    def __settings_to_pydantic_model(self) -> Self:
        """Convert sub-settings into Pydantic models."""
        if isinstance(self.gauss_seidel_settings, Mapping):
            self.gauss_seidel_settings = MDAGaussSeidel_Settings(
                **self.gauss_seidel_settings
            )
        if isinstance(self.newton_settings, Mapping):
            self.newton_settings = MDANewtonRaphson_Settings(**self.newton_settings)
        return self
