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

import os
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings

from gemseo.core.algorithm.base_driver_library import BaseDriverLibrary
from gemseo.core.discipline.base_discipline import BaseDiscipline
from gemseo.core.discipline.execution_statistics import ExecutionStatistics
from gemseo.core.discipline.execution_status import ExecutionStatus
from gemseo.core.function.preprocessed_function import PreprocessedFunction
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.mda.core.base_parallel_solver_settings import BaseMDAParallelSolverSettings
from gemseo.util._directory_manager.settings import Settings as DirectoryManagerSettings
from gemseo.util.constant import _check_desvars_bounds
from gemseo.util.constant import _enable_discipline_cache
from gemseo.util.constant import _enable_discipline_statistics
from gemseo.util.constant import _enable_discipline_status
from gemseo.util.constant import _enable_function_statistics
from gemseo.util.constant import _enable_parallel_execution
from gemseo.util.constant import _enable_progress_bar
from gemseo.util.constant import _validate_input_data
from gemseo.util.constant import _validate_output_data
from gemseo.util.constant import n_cpus
from gemseo.util.logging import LoggingConfiguration

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from typing import Any
    from typing import Self

    from pydantic import ValidationInfo


def _apply_check_desvars_bounds(value: bool) -> None:
    """Apply `check_desvars_bounds`.

    Args:
        value: The value of the field.
    """
    EvaluationProblem.check_bounds = value


def _apply_enable_discipline_cache(value: bool) -> None:
    """Apply `enable_discipline_cache`.

    Args:
        value: The value of the field.

    Note:
        [BaseMDA][gemseo.mda.core.base.BaseMDA] overrides `default_cache_type`
        on purpose: MDAs keep their cache.
    """
    BaseDiscipline.default_cache_type = (
        BaseDiscipline.CacheType.SIMPLE if value else BaseDiscipline.CacheType.NONE
    )


def _apply_enable_discipline_statistics(value: bool) -> None:
    """Apply `enable_discipline_statistics`.

    Args:
        value: The value of the field.
    """
    ExecutionStatistics.is_enabled = value


def _apply_enable_discipline_status(value: bool) -> None:
    """Apply `enable_discipline_status`.

    Args:
        value: The value of the field.
    """
    ExecutionStatus.is_enabled = value


def _apply_enable_function_statistics(value: bool) -> None:
    """Apply `enable_function_statistics`.

    Args:
        value: The value of the field.
    """
    PreprocessedFunction.enable_statistics = value


def _apply_enable_parallel_execution(value: bool) -> None:
    """Apply `enable_parallel_execution`.

    Args:
        value: The value of the field.
    """
    BaseMDAParallelSolverSettings.set_default_n_processes(n_cpus if value else 1)
    CallableParallelExecution.enable_parallel_execution = value


def _apply_enable_progress_bar(value: bool) -> None:
    """Apply `enable_progress_bar`.

    Args:
        value: The value of the field.
    """
    BaseDriverLibrary.enable_progress_bar = value


def _apply_validate_input_data(value: bool) -> None:
    """Apply `validate_input_data`.

    Args:
        value: The value of the field.
    """
    BaseDiscipline.validate_input_data = value


def _apply_validate_output_data(value: bool) -> None:
    """Apply `validate_output_data`.

    Args:
        value: The value of the field.
    """
    BaseDiscipline.validate_output_data = value


_apply: Final[Mapping[str, Callable[[bool], None]]] = MappingProxyType({
    "check_desvars_bounds": _apply_check_desvars_bounds,
    "enable_discipline_cache": _apply_enable_discipline_cache,
    "enable_discipline_statistics": _apply_enable_discipline_statistics,
    "enable_discipline_status": _apply_enable_discipline_status,
    "enable_function_statistics": _apply_enable_function_statistics,
    "enable_parallel_execution": _apply_enable_parallel_execution,
    "enable_progress_bar": _apply_enable_progress_bar,
    "validate_input_data": _apply_validate_input_data,
    "validate_output_data": _apply_validate_output_data,
})
"""The function applying a field to the class that it configures, per field name."""

_non_fast_field_names: Final[frozenset[str]] = frozenset({"enable_progress_bar"})
"""The names of the applied fields that the fast mode does not disable.

The progress bar is user feedback and not a per-evaluation overhead.
"""

_fast_field_names: Final[frozenset[str]] = frozenset(
    _apply.keys() - _non_fast_field_names
)
"""The names of the fields disabled by the fast mode."""

_fast_removed_message: Final[str] = (
    "The fast option of the global configuration has been removed; "
    "use its methods enable_fast_mode and disable_fast_mode instead, "
    "and unset the GEMSEO_FAST environment variable if it is set."
)
"""The error message raised when the removed `fast` option is passed."""


class GlobalConfiguration(
    BaseSettings,
    validate_assignment=True,
    env_nested_delimiter="_",
    env_nested_max_split=1,
    env_prefix="GEMSEO_",
    env_file=".env",
    extra="ignore",
):  # noqa: N801
    """Global configuration."""

    check_desvars_bounds: bool = Field(
        default=_check_desvars_bounds,
        description="""Whether to check the membership of design variables in the bounds
when evaluating the functions
in [EvaluationProblem][gemseo.core.problem.evaluation.EvaluationProblem].""",
    )

    enable_discipline_cache: bool = Field(
        default=_enable_discipline_cache,
        description="Whether to enable the discipline cache.",
    )

    enable_discipline_statistics: bool = Field(
        default=_enable_discipline_statistics,
        description="""Whether to record execution statistics
of the disciplines such as
the execution time, the number of executions and the number of linearizations.""",
    )

    enable_discipline_status: bool = Field(
        default=_enable_discipline_status,
        description="Whether to enable discipline statuses.",
    )

    enable_function_statistics: bool = Field(
        default=_enable_function_statistics,
        description="""Whether to record the statistics attached to the functions,
in charge of counting their number of evaluations.""",
    )

    enable_parallel_execution: bool = Field(
        default=_enable_parallel_execution,
        description="""Whether to let GEMSEO use parallelism
    (multi-processing or multi-threading) by default.

When `False`,
the default number of threads/processes of the parallel MDAs,
of
[ParallelDisciplineChain][gemseo.discipline.chain.parallel_chain.ParallelDisciplineChain]
and of
[CallableParallelExecution][gemseo.core.parallel_execution.callable_parallel_execution.CallableParallelExecution]
is 1.""",
    )

    enable_progress_bar: bool = Field(
        default=_enable_progress_bar,
        description="""Whether to enable the progress bar attached to the drivers,
in charge to log the execution of the process:
iteration, execution time and objective value.""",
    )

    validate_input_data: bool = Field(
        default=_validate_input_data,
        description="""Whether to validate the input data of a discipline
before execution.""",
    )

    validate_output_data: bool = Field(
        default=_validate_output_data,
        description="""Whether to validate the output data of a discipline
after execution.""",
    )

    logging: LoggingConfiguration = Field(
        default=LoggingConfiguration(),
        description=LoggingConfiguration.__doc__,
    )

    directory_manager: DirectoryManagerSettings = Field(
        default=DirectoryManagerSettings(),
        description=DirectoryManagerSettings.__doc__,
    )

    def __init__(self, **data: Any) -> None:
        """
        Args:
            **data: The values of the fields.

        Raises:
            ValueError: When the `fast` option is passed.
        """  # noqa: D205 D212
        # Only the constructor spells the option without the prefix;
        # a prefix-free name coming from a dotenv file
        # belongs to another tool and is ignored.
        if "fast" in {name.lower() for name in data}:
            raise ValueError(_fast_removed_message)

        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def __reject_fast(cls, data: Any) -> Any:
        """Reject the `fast` option, replaced by the fast mode methods.

        Args:
            data: The input data.

        Returns:
            The input data.

        Raises:
            ValueError: When the `fast` option is set
                in the environment or in a dotenv file.
        """
        # A dotenv file passes an unknown key with its prefix,
        # whereas the environment source only reads the names of the fields.
        fast_name = f"{cls.model_config['env_prefix']}fast".lower()
        names = [*data, *os.environ] if isinstance(data, dict) else list(os.environ)
        if any(isinstance(name, str) and name.lower() == fast_name for name in names):
            raise ValueError(_fast_removed_message)

        return data

    @model_validator(mode="after")
    def __apply_fields(self, info: ValidationInfo) -> Self:
        """Apply the fields to the classes that they configure.

        All the fields are applied when the global configuration is created,
        and only the field being assigned otherwise,
        so that an assignment does not overwrite a class attribute
        that has been set directly.

        Args:
            info: The validation context,
                holding the name of the field being assigned,
                which is `None` when the global configuration is being created.

        Returns:
            The global configuration.
        """
        field_name = info.field_name
        if field_name is None:
            field_names = _apply
        elif field_name in _apply:
            field_names = (field_name,)
        else:
            field_names = ()

        for name in field_names:
            _apply[name](getattr(self, name))

        return self

    def enable_fast_mode(self) -> None:
        """Configure GEMSEO for inexpensive disciplines.

        Disable the options that cost time at every evaluation:

        - `check_desvars_bounds`,
        - `enable_discipline_cache`,
        - `enable_discipline_statistics`,
        - `enable_discipline_status`,
        - `enable_function_statistics`,
        - `enable_parallel_execution`,
        - `validate_input_data`,
        - `validate_output_data`.

        `enable_progress_bar` is deliberately left alone,
        as the progress bar is user feedback and not a per-evaluation overhead.

        The cache of the MDAs is not disabled either,
        as an MDA without cache re-executes its disciplines
        at every residual evaluation.

        These options are applied again at every call,
        which restores the fast mode
        after one of the class attributes that they drive
        has been changed directly.
        """
        self.__set_fast_mode(True)

    def disable_fast_mode(self) -> None:
        """Reset the options disabled by the fast mode to their default values.

        The options enabled by default are enabled again;
        `enable_discipline_statistics`,
        `enable_discipline_status`
        and `enable_function_statistics`,
        which are disabled by default,
        remain disabled.

        These default values are the built-in ones,
        so an option set from an environment variable or from a dotenv file
        is reset to the built-in default too,
        and not to the value read at start-up.
        """
        self.__set_fast_mode(False)

    def __set_fast_mode(self, enable: bool) -> None:
        """Set the options driven by the fast mode.

        Args:
            enable: Whether to enable the fast mode.
        """
        fields = type(self).model_fields
        for name in _fast_field_names:
            setattr(self, name, False if enable else fields[name].default)


_configuration = GlobalConfiguration()
"""The global GEMSEO configuration.

The feature is described
on the page [Global configuration][concept-global-configuration] of the user guide.
"""
