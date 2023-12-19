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
"""Parallel execution of disciplines."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline import MDODiscipline
    from gemseo.core.discipline_data import Data


class DiscParallelExecution(CallableParallelExecution):
    """Execute disciplines in parallel."""

    _disciplines: Sequence[MDODiscipline]
    """The disciplines to execute."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        n_processes: int = CallableParallelExecution.N_CPUS,
        use_threading: bool = False,
        wait_time_between_fork: float = 0.0,
        exceptions_to_re_raise: tuple[type[Exception]] = (),
    ) -> None:
        """
        Args:
            disciplines: The disciplines to execute.
        """  # noqa:D205 D212 D415
        super().__init__(
            workers=[d.execute for d in disciplines],
            n_processes=n_processes,
            use_threading=use_threading,
            wait_time_between_fork=wait_time_between_fork,
            exceptions_to_re_raise=exceptions_to_re_raise,
        )
        # Because accessing a method of an object provides a new callable object for
        # every access, we shall check unicity on the disciplines.
        self._check_unicity(disciplines)
        self._disciplines = disciplines

    def execute(  # noqa: D102
        self,
        inputs: Sequence[Data | None],
        exec_callback: Callable[[int, Any], Any] | None = None,
        task_submitted_callback: Callable | None = None,
    ) -> list[Any]:
        ordered_outputs = super().execute(
            inputs,
            exec_callback=exec_callback,
            task_submitted_callback=task_submitted_callback,
        )

        if len(self._disciplines) == 1 or len(self._disciplines) != len(inputs):
            if (
                not self.use_threading
                and self.MULTI_PROCESSING_START_METHOD
                == self.MultiProcessingStartMethod.SPAWN
            ):
                self._disciplines[0].n_calls += len(inputs)
        else:
            for disc, output in zip(self._disciplines, ordered_outputs):
                # When the discipline in the worker failed, output is None.
                # We do not update the local_data such that the issue is caught by the
                # output grammar.
                if output is not None:
                    disc.local_data = output

        return ordered_outputs
