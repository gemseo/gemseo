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
"""Base settings class for parallel MDA algorithms."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic import NonNegativeInt

from gemseo.mda.core.base_solver_settings import BaseMDASolverSettings
from gemseo.util.constant import _enable_parallel_execution
from gemseo.util.constant import n_cpus


class BaseMDAParallelSolverSettings(BaseMDASolverSettings):
    """The settings for the MDA algorithms that can be run in parallel."""

    _default_n_processes: ClassVar[int] = n_cpus if _enable_parallel_execution else 1
    """The default number of threads/processes.

    This default is shared by all the settings classes deriving from this one,
    except those that override the default of `n_processes`,
    whether by redeclaring the field
    or by listing it in `_inherited_field_defaults`.
    """

    execute_before_linearizing: bool = Field(
        default=True,
        description="""Whether to start by executing the disciplines before linearizing.
            This ensures that the discipline are executed and linearized with the same
            input data. It can be almost free if the corresponding output data have been
            stored in the [BaseMDA.cache][gemseo.mda.core.base.BaseMDA.cache].""",
    )

    n_processes: NonNegativeInt = Field(
        default_factory=lambda: BaseMDAParallelSolverSettings._default_n_processes,
        description="""The number of threads/processes.

Threads if `use_threading`, processes otherwise.

The default value can be changed
using
[set_default_n_processes()][gemseo.mda.core.base_parallel_solver_settings.BaseMDAParallelSolverSettings.set_default_n_processes],
or by the `enable_parallel_execution` option of the global configuration.
""",
    )

    use_threading: bool = Field(
        default=True,
        description=(
            """Whether to use threads instead of processes to parallelize the execution.

Processes will copy (serialize) the disciplines, while threads will share the memory.
If one wants to execute the same discipline multiple times,
then multiprocessing should be preferred."""
        ),
    )

    @classmethod
    def set_default_n_processes(cls, default_n_processes: int) -> None:
        """Set the default number of threads/processes.

        The field reads this default when it is validated,
        so this default is shared
        by all the settings classes deriving from
        [BaseMDAParallelSolverSettings][gemseo.mda.core.base_parallel_solver_settings.BaseMDAParallelSolverSettings],
        whatever the class that this classmethod is called on,
        including the settings models embedded in other settings models.
        A settings class overriding the default of `n_processes`,
        whether by redeclaring the field
        or by listing it in `_inherited_field_defaults`,
        keeps its own default.

        Args:
            default_n_processes: The default number of threads/processes.
        """
        BaseMDAParallelSolverSettings._default_n_processes = default_n_processes
