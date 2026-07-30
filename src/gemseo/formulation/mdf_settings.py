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

from pydantic import Field

from gemseo.formulation.core.base_settings import BaseFormulationSettings
from gemseo.mda.chain_settings import MDAChain_Settings
from gemseo.mda.core.base_settings import BaseMDASettings  # noqa: TC001


class MDF_Settings(BaseFormulationSettings):  # noqa: N801
    """Settings of the [MDF][gemseo.formulation.mdf.MDF] formulation."""

    main_mda_settings: BaseMDASettings = Field(
        default_factory=MDAChain_Settings,
        description="""The settings of the main MDA.

These settings may include those of the inner-MDA.""",
    )
