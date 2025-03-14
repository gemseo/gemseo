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

from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.utils.name_generator import NameGenerator


class BiLevel_Settings(MDF_Settings):  # noqa: N801
    """Settings of the :class:`.BiLevel` formulation."""

    _TARGET_CLASS_NAME = "BiLevel"

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

    keep_opt_history: bool = Field(
        default=True,
        description="""Whether to keep database copies of the sub-scenario adapters
after each execution.
Depending on the size of the databases and the number of consecutive
executions, this can be very memory consuming. If the adapter will be executed in
parallel, the databases will not be saved to the main process by the
sub-processes, so this setting should be set to ``False`` to avoid
unnecessary memory use in the sub-processes.""",
    )

    save_opt_history: bool = Field(
        default=False,
        description="""Whether to save the optimization history
to an HDF5 file after each execution.""",
    )

    naming: NameGenerator.Naming = Field(
        default=NameGenerator.Naming.NUMBERED,
        description="""The way of naming the database files.
When the adapter will be executed in parallel, this method shall be set
to ``UUID`` because this method is multiprocess-safe.""",
    )
