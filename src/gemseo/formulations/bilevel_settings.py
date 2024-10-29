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
"""Settings of the BiLevel formulation ."""

from __future__ import annotations

from pydantic import Field

from gemseo.formulations.mdf_settings import MDFSettings


class BiLevelSettings(MDFSettings):
    """Settings of the :class:`.BiLevel` formulation."""

    parallel_scenarios: bool = Field(
        default=False, description="Whether to run the sub-scenarios in parallel."
    )

    multithread_scenarios: bool = Field(
        default=True,
        description="""If ``True`` and parallel_scenarios=True,
        the sub-scenarios are run in parallel using multi-threading;
        if False and parallel_scenarios=True, multiprocessing is used.""",
    )

    apply_cstr_tosub_scenarios: bool = Field(
        default=True,
        description="""Whether the :meth:`.add_constraint` method
        adds the constraint to the optimization problem of the sub-scenario
        capable of computing the constraint.""",
    )

    apply_cstr_to_system: bool = Field(
        default=True,
        description="""Whether the :meth:`.add_constraint` method adds
        the constraint to the optimization problem of the system scenario.""",
    )

    reset_x0_before_opt: bool = Field(
        default=False,
        description="""Whether to restart the sub optimizations
        from the initial guesses, otherwise warm start them.""",
    )

    sub_scenarios_log_level: int | None = Field(
        default=None,
        description="""The level of the root logger
        during the sub-scenarios executions.
        If ``None``, do not change the level of the root logger.""",
    )
