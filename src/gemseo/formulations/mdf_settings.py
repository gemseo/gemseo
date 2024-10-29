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

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator

from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.mda_chain import MDAChain
from gemseo.typing import StrKeyMapping  # noqa: TCH001

if TYPE_CHECKING:
    from typing_extensions import Self


class MDFSettings(BaseFormulationSettings):
    """Settings of the :class:`.MDF` formulation."""

    main_mda_name: str = Field(
        default=MDAChain.__name__,
        description="""The name of the class of the main MDA.

        Typically the :class:`.MDAChain`,
        but one can force to use :class:`.MDAGaussSeidel` for instance.""",
    )

    inner_mda_name: str = Field(
        default=MDAJacobi.__name__,
        description="""The name of the class of the inner MDA if any.

        Typically when the main MDA is an :class:`.MDAChain`.""",
    )

    main_mda_settings: StrKeyMapping = Field(
        default_factory=dict,
        description="""The settings of the main MDA.

        These settings may include those of the inner-MDA.""",
    )

    @model_validator(mode="after")
    def __update_main_mda_settings(self) -> Self:
        """Update the ``main_mda_settings`` with ``model_extra``."""
        if self.model_extra:
            self.main_mda_settings = dict(self.main_mda_settings)
            self.main_mda_settings.update(self.model_extra)

        if self.main_mda_name == MDAChain.__name__:
            self.main_mda_settings["inner_mda_name"] = self.inner_mda_name
        return self
