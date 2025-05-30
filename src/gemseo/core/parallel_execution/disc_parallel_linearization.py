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
"""Parallel execution of linearized disciplines."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple

from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.core.parallel_execution.callable_parallel_execution import CallbackType
from gemseo.typing import StrKeyMapping
from gemseo.utils.constants import N_CPUS

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.typing import JacobianData


class _WorkerData(NamedTuple):
    """The computed data of a worker (discipline)."""

    io_data: DisciplineData
    jacobian: JacobianData


class _Functor:
    """A functor to call a discipline linearization.

    When called, the :attr:`.Discipline.io.data` and :attr:`.Discipline.jac`
    are returned.
    """

    def __init__(self, discipline: Discipline, execute: bool = True) -> None:
        """
        Args:
            discipline: The discipline to get a callable from.
            execute: Whether to start by executing the discipline
                with the input data for which to compute the Jacobian;
                this allows to ensure that the discipline was executed
                with the right input data;
                it can be almost free if the corresponding output data
                have been stored in the :attr:`.cache`.
        """  # noqa:D205 D212 D415
        self.__disc = discipline
        self.__execute = execute

    def __call__(self, inputs: StrKeyMapping) -> _WorkerData:
        """
        Args:
            inputs: The inputs of the discipline.

        Returns:
            The discipline :attr:`.Discipline.io.data` and its jacobian.
        """  # noqa:D205 D212 D415
        jacobian = self.__disc.linearize(inputs, execute=self.__execute)
        return _WorkerData(self.__disc.io.data, jacobian)


class DiscParallelLinearization(CallableParallelExecution[StrKeyMapping, _WorkerData]):
    """Linearize disciplines in parallel."""

    _disciplines: Sequence[Discipline]
    """The disciplines to linearize."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        n_processes: int = N_CPUS,
        use_threading: bool = False,
        wait_time_between_fork: float = 0.0,
        exceptions_to_re_raise: Sequence[type[Exception]] = (),
        execute: bool = True,
    ) -> None:
        """
        Args:
            disciplines: The disciplines to execute.
            execute: Whether to start by executing the discipline
                with the input data for which to compute the Jacobian;
                this allows to ensure that the discipline was executed
                with the right input data;
                it can be almost free if the corresponding output data
                have been stored in the :attr:`.cache`.
        """  # noqa:D205 D212 D415
        super().__init__(
            workers=[_Functor(d, execute=execute) for d in disciplines],
            n_processes=n_processes,
            use_threading=use_threading,
            wait_time_between_fork=wait_time_between_fork,
            exceptions_to_re_raise=exceptions_to_re_raise,
        )
        # Because accessing a method of an object provides a new callable object for
        # every access, we shall check unicity on the disciplines.
        self._check_unicity(disciplines)
        self._disciplines = disciplines

    # TODO: API: fix return type or return None and use the disc attributes updated?
    def execute(  # type: ignore[override] # noqa: D102
        self,
        inputs: Sequence[StrKeyMapping],
        exec_callback: CallbackType | Iterable[CallbackType] = (),
        task_submitted_callback: Callable[[], None] | None = None,
    ) -> list[JacobianData | None]:
        ordered_outputs = super().execute(
            inputs,
            exec_callback=exec_callback,
            task_submitted_callback=task_submitted_callback,
        )

        if len(self._disciplines) == 1 or len(self._disciplines) != len(inputs):
            output_0 = ordered_outputs[0]
            if output_0 is not None:
                disc_0 = self._disciplines[0]
                if len(self._disciplines) == 1:
                    disc_0.io.data = output_0.io_data
                    disc_0.jac = output_0.jacobian
                if (
                    not self.use_threading
                    and self.MULTI_PROCESSING_START_METHOD
                    == self.MultiProcessingStartMethod.SPAWN
                    and ExecutionStatistics.is_enabled
                    and output_0.io_data
                ):
                    # Only increase the number of calls if the Jacobian was computed.
                    disc_0.execution_statistics.n_executions += len(inputs)  # type: ignore[operator] # checked with activate_counter
                    disc_0.execution_statistics.n_linearizations += len(inputs)  # type: ignore[operator] # checked with activate_counter
        else:
            for disc, output in zip(self._disciplines, ordered_outputs):
                # When the discipline in the worker failed, output is None.
                # We do not update the data such that the issue is caught by the
                # output grammar.
                if output is not None:
                    disc.io.data = output.io_data
                    disc.jac = output.jacobian

        return [out.jacobian for out in ordered_outputs if out is not None or None]
