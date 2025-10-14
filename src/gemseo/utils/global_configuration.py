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

"""Global GEMSEO configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic_settings import BaseSettings

from gemseo.algos.base_driver_library import BaseDriverLibrary
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.problem_function import ProblemFunction
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.core.execution_status import ExecutionStatus
from gemseo.mda.base_parallel_mda_settings import BaseParallelMDASettings
from gemseo.utils.constants import _CHECK_DESVARS_BOUNDS
from gemseo.utils.constants import _ENABLE_DISCIPLINE_CACHE
from gemseo.utils.constants import _ENABLE_DISCIPLINE_STATISTICS
from gemseo.utils.constants import _ENABLE_DISCIPLINE_STATUS
from gemseo.utils.constants import _ENABLE_FUNCTION_STATISTICS
from gemseo.utils.constants import _ENABLE_PARALLEL_EXECUTION
from gemseo.utils.constants import _ENABLE_PROGRESS_BAR
from gemseo.utils.constants import _VALIDATE_INPUT_DATA
from gemseo.utils.constants import _VALIDATE_OUTPUT_DATA
from gemseo.utils.constants import N_CPUS
from gemseo.utils.logging import LoggingConfiguration

if TYPE_CHECKING:
    from typing_extensions import Self


class GlobalConfiguration(
    BaseSettings,
    validate_assignment=True,
    env_nested_delimiter="_",
    env_prefix="GEMSEO_",
    env_file=".env",
    extra="ignore",
):  # noqa: N801
    """Global configuration."""

    check_desvars_bounds: bool = Field(
        default=_CHECK_DESVARS_BOUNDS,
        description="""Whether to check the membership of design variables in the bounds
when evaluating the functions in :class:`.OptimizationProblem`.""",
    )

    enable_discipline_cache: bool = Field(
        default=_ENABLE_DISCIPLINE_CACHE,
        description="Whether to enable the discipline cache.",
    )

    enable_discipline_statistics: bool = Field(
        default=_ENABLE_DISCIPLINE_STATISTICS,
        description="""Whether to record execution statistics
of the disciplines such as
the execution time, the number of executions and the number of linearizations.""",
    )

    enable_discipline_status: bool = Field(
        default=_ENABLE_DISCIPLINE_STATUS,
        description="Whether to enable discipline statuses.",
    )

    enable_function_statistics: bool = Field(
        default=_ENABLE_FUNCTION_STATISTICS,
        description="""Whether to record the statistics attached to the functions,
in charge of counting their number of evaluations.""",
    )

    enable_parallel_execution: bool = Field(
        default=_ENABLE_PARALLEL_EXECUTION,
        description="""Whether to let |g| use parallelism
    (multi-processing or multi-threading) by default.""",
    )

    enable_progress_bar: bool = Field(
        default=_ENABLE_PROGRESS_BAR,
        description="""Whether to enable the progress bar attached to the drivers,
in charge to log the execution of the process:
iteration, execution time and objective value.""",
    )

    fast: bool = Field(
        default=False,
        description="""Use a global configuration for inexpensive disciplines.

    This global configuration disables the following options:

    - ``check_desvars_bounds``,
    - ``enable_discipline_cache``,
    - ``enable_discipline_statistics``,
    - ``enable_discipline_status``,
    - ``enable_parallel_execution``,
    - ``validate_input_data``,
    - ``validate_output_data``.
    """,
    )

    logging: LoggingConfiguration = Field(
        default=LoggingConfiguration(), description="The logging configuration."
    )

    validate_input_data: bool = Field(
        default=_VALIDATE_INPUT_DATA,
        description="""Whether to validate the input data of a discipline
before execution.""",
    )

    validate_output_data: bool = Field(
        default=_VALIDATE_OUTPUT_DATA,
        description="""Whether to validate the output data of a discipline
after execution.""",
    )

    @field_validator("check_desvars_bounds")
    @classmethod
    def __validate_check_desvars_bounds(cls, v: bool) -> bool:
        OptimizationProblem.check_bounds = v
        return v

    @field_validator("enable_discipline_cache")
    @classmethod
    def __validate_enable_discipline_cache(cls, v: bool) -> bool:
        Discipline.default_cache_type = (
            Discipline.CacheType.SIMPLE if v else Discipline.CacheType.NONE
        )
        return v

    @field_validator("enable_discipline_statistics")
    @classmethod
    def __validate_enable_discipline_statistics(cls, v: bool) -> bool:
        ExecutionStatistics.is_enabled = v
        return v

    @field_validator("enable_discipline_status")
    @classmethod
    def __validate_enable_discipline_status(cls, v: bool) -> bool:
        ExecutionStatus.is_enabled = v
        return v

    @field_validator("enable_function_statistics")
    @classmethod
    def __validate_enable_function_statistics(cls, v: bool) -> bool:
        ProblemFunction.enable_statistics = v
        return v

    @field_validator("enable_parallel_execution")
    @classmethod
    def __validate_enable_parallel_execution(cls, v: bool) -> bool:
        default_n_processes = N_CPUS if v else 1
        BaseParallelMDASettings.set_default_n_processes(default_n_processes)
        return v

    @field_validator("enable_progress_bar")
    @classmethod
    def __validate_enable_progress_bar(cls, v: bool) -> bool:
        BaseDriverLibrary.enable_progress_bar = v
        return v

    @field_validator("validate_input_data")
    @classmethod
    def __validate_validate_input_data(cls, v: bool) -> bool:
        Discipline.validate_input_data = v
        return v

    @field_validator("validate_output_data")
    @classmethod
    def __validate_validate_output_data(cls, v: bool) -> bool:
        Discipline.validate_output_data = v
        return v

    @model_validator(mode="after")
    def __validate_fast(self) -> Self:
        # setattr would validate the field and lead to a RecursionError;
        # so we use object.__setattr__ instead.
        if self.fast:
            for name in (
                "check_desvars_bounds",
                "enable_discipline_cache",
                "enable_discipline_statistics",
                "enable_discipline_status",
                "enable_function_statistics",
                "enable_parallel_execution",
                "validate_input_data",
                "validate_output_data",
            ):
                object.__setattr__(self, name, False)
        elif "fast" in self.model_fields_set:
            # The user sets fast to False.
            # If fast is at False by default,
            # do not do this,
            # as you would overwrite the other field values the user have entered.
            for name in (
                "check_desvars_bounds",
                "enable_discipline_cache",
                "enable_parallel_execution",
                "validate_input_data",
                "validate_output_data",
            ):
                object.__setattr__(self, name, True)

            for name in (
                "enable_discipline_statistics",
                "enable_discipline_status",
                "enable_function_statistics",
            ):
                object.__setattr__(self, name, False)

        return self


_configuration = GlobalConfiguration()
"""The global |g| configuration.

The feature is described on page :ref:`global_configuration` of the user guide.
"""
