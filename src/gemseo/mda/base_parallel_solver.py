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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base module for MDA solvers that can be run in parallel."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.parallel_execution.discipline_execution import DiscParallelExecution
from gemseo.core.parallel_execution.discipline_linearization import (
    DiscParallelLinearization,
)
from gemseo.mda.base_parallel_solver_settings import BaseMDAParallelSolverSettings
from gemseo.mda.base_solver import BaseMDASolver

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.typing import StrKeyMapping


class BaseMDAParallelSolver(BaseMDASolver):
    """Abstract class for MDA solvers that can be run in parallel."""

    settings_class: ClassVar[type[BaseMDAParallelSolverSettings]] = (
        BaseMDAParallelSolverSettings
    )
    """The pydantic model for the settings."""

    settings: BaseMDAParallelSolverSettings
    """The settings of the MDA"""

    __parallel_execution: DiscParallelExecution | None
    """Either a parallel executor for disciplines or `None` in serial mode."""

    __parallel_linearization: DiscParallelLinearization | None
    """Either a parallel linearizator for disciplines or `None` in serial mode."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        settings: BaseMDAParallelSolverSettings | None = None,
    ) -> None:
        super().__init__(disciplines, settings=settings)
        if self.settings.n_processes > 1:
            self._execute_disciplines = self._execute_disciplines_in_parallel
            self._linearize_disciplines = self._linearize_disciplines_in_parallel
            self.__parallel_execution = DiscParallelExecution(
                disciplines,
                self.settings.n_processes,
                use_threading=self.settings.use_threading,
                exceptions_to_re_raise=(ValueError,),
            )
            self.__parallel_linearization = DiscParallelLinearization(
                disciplines,
                self.settings.n_processes,
                use_threading=self.settings.use_threading,
                execute=self.settings.execute_before_linearizing,
            )
        else:
            self._execute_disciplines = self._execute_disciplines_sequentially
            self._linearize_disciplines = self._linearize_disciplines_sequentially
            self.__parallel_execution = None
            self.__parallel_linearization = None

        self._compute_input_coupling_names()

    @property
    def parallel_execution(self) -> DiscParallelExecution | None:
        """The parallel executor, if any."""
        return self.__parallel_execution

    def _execute_disciplines_sequentially(self, input_data: StrKeyMapping) -> None:
        """Execute the discipline sequentially.

        Args:
            input_data: The input data to execute the disciplines on.
        """
        for discipline in self._disciplines:
            discipline.execute(input_data)

    def _execute_disciplines_in_parallel(self, input_data: StrKeyMapping) -> None:
        """Execute the discipline in parallel.

        Args:
            input_data: The input data to execute the disciplines on.
        """
        self.__parallel_execution.execute([input_data] * len(self._disciplines))

    def _linearize_disciplines_sequentially(self, input_data: StrKeyMapping) -> None:
        """Linearize the disciplines sequentially.

        Args:
            input_data: The input data defining the point around which the disciplines
                are linearize.
        """
        execute_before_linearizing = self.settings.execute_before_linearizing
        for discipline in self._disciplines:
            discipline.linearize(input_data, execute=execute_before_linearizing)

    def _linearize_disciplines_in_parallel(self, input_data: StrKeyMapping) -> None:
        """Linearize the disciplines in parallel.

        Args:
            input_data: The input data defining the point around which the disciplines
                are linearize.
        """
        self.__parallel_linearization.execute([input_data] * len(self._disciplines))

    def _execute_disciplines_and_update_local_data(self) -> None:
        self._execute_disciplines(self.io.input_data)

        for discipline in self._disciplines:
            output_data = discipline.io.output_data
            self.io.output_data |= output_data
            self.io.propagate_to_input(output_data)

        self._compute_name_to_slice()
