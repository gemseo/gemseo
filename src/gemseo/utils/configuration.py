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

"""GEMSEO configuration settings."""

from __future__ import annotations

from functools import partial

from pydantic import Field

from gemseo.settings.base_settings import BaseSettings
from gemseo.utils.pydantic import copy_field


class GEMSEO_Settings(BaseSettings):  # noqa: N801
    """GEMSEO configuration settings."""

    check_desvars_bounds: bool = Field(
        default=True,
        description="""Whether to check the membership of design variables in the bounds
when evaluating the functions in OptimizationProblem.""",
    )

    enable_discipline_cache: bool = Field(
        default=True, description="Whether to enable the discipline cache."
    )

    enable_discipline_statistics: bool = Field(
        default=False,
        description="""Whether to record execution statistics
of the disciplines such as
the execution time, the number of executions and the number of linearizations.""",
    )

    enable_discipline_status: bool = Field(
        default=False, description="Whether to enable discipline statuses."
    )

    enable_function_statistics: bool = Field(
        default=False,
        description="""Whether to record the statistics attached to the functions,
in charge of counting their number of evaluations.""",
    )

    enable_parallel_execution: bool = Field(
        default=True,
        description="""Whether to let |g| use parallelism
    (multi-processing or multi-threading) by default.""",
    )

    enable_progress_bar: bool = Field(
        default=True,
        description="""Whether to enable the progress bar attached to the drivers,
in charge to log the execution of the process:
iteration, execution time and objective value.""",
    )

    validate_input_data: bool = Field(
        default=True,
        description="""Whether to validate the input data of a discipline
before execution.""",
    )

    validate_output_data: bool = Field(
        default=True,
        description="""Whether to validate the output data of a discipline
after execution.""",
    )


_copy_field = partial(copy_field, model=GEMSEO_Settings)


class Fast_GEMSEO_Settings(GEMSEO_Settings):  # noqa: N801
    """GEMSEO configuration settings preconfigured for inexpensive disciplines.

    The following options are disabled:

    - discipline cache,
    - parallel execution,
    - verification of design variable bounds,
    - validation of discipline input data,
    - validation of discipline output data.
    """

    check_desvars_bounds: bool = _copy_field("check_desvars_bounds", default=False)
    enable_discipline_cache: bool = _copy_field(
        "enable_discipline_cache", default=False
    )
    enable_parallel_execution: bool = _copy_field(
        "enable_parallel_execution", default=False
    )
    validate_input_data: bool = _copy_field("validate_input_data", default=False)
    validate_output_data: bool = _copy_field("validate_output_data", default=False)
