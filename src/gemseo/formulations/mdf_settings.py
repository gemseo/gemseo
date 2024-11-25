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
"""Settings of the MDF formulation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator

from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
from gemseo.mda.base_mda_settings import BaseMDASettings  # noqa: TC001
from gemseo.mda.factory import MDAFactory
from gemseo.mda.mda_chain import MDAChain
from gemseo.typing import StrKeyMapping  # noqa: TC001

if TYPE_CHECKING:
    from typing_extensions import Self


class MDF_Settings(BaseFormulationSettings):  # noqa: N801
    """Settings of the :class:`.MDF` formulation."""

    _TARGET_CLASS_NAME = "MDF"

    main_mda_name: str = Field(
        default=MDAChain.__name__,
        description="""The name of the class of the main MDA.

Typically the :class:`.MDAChain`,
but one can force to use :class:`.MDAGaussSeidel` for instance.""",
    )

    main_mda_settings: StrKeyMapping | BaseMDASettings = Field(
        default_factory=dict,
        description="""The settings of the main MDA.

These settings may include those of the inner-MDA.""",
    )

    @model_validator(mode="after")
    def __validate_mda_settings(self) -> Self:
        """Validate the main MDA settings using the appropriate Pydantic model."""
        settings_model = MDAFactory().get_class(self.main_mda_name).Settings
        if isinstance(self.main_mda_settings, Mapping):
            self.main_mda_settings = settings_model(**self.main_mda_settings)
        if not isinstance(self.main_mda_settings, settings_model):
            msg = (
                f"The {self.main_mda_name} settings model has the wrong type: "
                f"expected {settings_model.__name__}, "
                f"got {self.main_mda_settings.__class__.__name__}."
            )
            raise TypeError(msg)
        return self
