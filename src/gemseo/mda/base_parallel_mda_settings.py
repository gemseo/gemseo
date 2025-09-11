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

from pydantic import Field
from pydantic import NonNegativeInt

from gemseo.mda.base_mda_solver_settings import BaseMDASolverSettings
from gemseo.utils.constants import _ENABLE_PARALLEL_EXECUTION
from gemseo.utils.constants import N_CPUS


class BaseParallelMDASettings(BaseMDASolverSettings):
    """The settings for the MDA algorithms that can be run in parallel."""

    execute_before_linearizing: bool = Field(
        default=True,
        description="""Whether to start by executing the disciplines before linearizing.
            This ensures that the discipline are executed and linearized with the same
            input data. It can be almost free if the corresponding output data have been
            stored in the :attr:`.BaseMDA.cache`.""",
    )

    n_processes: NonNegativeInt = Field(
        default=N_CPUS if _ENABLE_PARALLEL_EXECUTION else 1,
        description="""The number of threads/processes.

Threads if ``use_threading``, processes otherwise.

The default value can be changed
using :meth:`.set_default_n_processes` or :func:`.configure`.
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

        Args:
            default_n_processes: The default number of threads/processes.
        """
        try:
            fields = cls.__pydantic_fields__
        except AttributeError:  # pragma: no cover
            # TODO: remove when pydantic 2.9 is no longer supported.
            fields = cls.model_fields
        fields["n_processes"].default = default_n_processes
        cls.model_rebuild(force=True)
