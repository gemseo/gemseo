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
"""Settings of the IDF formulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import PositiveInt
from pydantic import model_validator

from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
from gemseo.typing import StrKeyMapping  # noqa: TCH001

if TYPE_CHECKING:
    from typing_extensions import Self


class IDFSettings(BaseFormulationSettings):
    """Settings of the :class:`.IDF` formulation."""

    normalize_constraints: bool = Field(
        default=True,
        description=(
            "Whether the outputs of the coupling consistency constraints are scaled."
        ),
    )

    n_processes: PositiveInt = Field(
        default=1,
        description="""The maximum simultaneous number of threads
        if ``use_threading`` is True, or processes otherwise,
        used to parallelize the execution.""",
    )

    use_threading: bool = Field(
        default=True,
        description="""Whether to use threads instead of processes
        to parallelize the execution;
        multiprocessing will copy (serialize) all the disciplines,
        while threading will share all the memory.
        This is important to note
        if you want to execute the same discipline multiple times,
        you shall use multiprocessing.""",
    )

    start_at_equilibrium: bool = Field(
        default=False,
        description="Whether an MDA is used to initialize the coupling variables.",
    )

    mda_options_for_start_at_equilibrium: StrKeyMapping = Field(
        default_factory=dict,
        description="""The options for the MDA when ``start_at_equilibrium=True``.

        See detailed options in :class:`.MDAChain`.""",
    )

    @model_validator(mode="after")
    def __update_main_mda_settings(self) -> Self:
        """Update the ``main_mda_settings`` with ``model_extra``."""
        if self.model_extra:
            self.mda_options_for_start_at_equilibrium = dict(
                self.mda_options_for_start_at_equilibrium
            )
            self.mda_options_for_start_at_equilibrium.update(self.model_extra)

        return self