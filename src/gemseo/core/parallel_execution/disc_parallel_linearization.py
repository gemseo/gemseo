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

from typing import Any
from typing import Callable
from typing import Sequence

from numpy import ndarray

from gemseo.core.discipline import MDODiscipline
from gemseo.core.discipline_data import Data
from gemseo.core.discipline_data import DisciplineData
from gemseo.core.parallel_execution.callable_parallel_execution import IS_WIN
from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution


class _Functor:
    """A functor to call a discipline linearization.

    When called, the :attr:`.MDODiscipline.local_data` and :attr:`.MDODiscipline.jac`
    are returned.
    """

    def __init__(self, discipline: MDODiscipline) -> None:
        """
        Args:
            discipline: The discipline to get a callable from.
        """  # noqa:D205 D212 D415
        self.__disc = discipline

    def __call__(
        self, inputs: Data | None
    ) -> tuple[DisciplineData, dict[str, dict[str, ndarray]]]:
        """
        Args:
            inputs: The inputs of the discipline.

        Returns:
            The discipline :attr:`.MDODiscipline.local_data` and its jacobian.
        """  # noqa:D205 D212 D415
        jac = self.__disc.linearize(inputs)
        return self.__disc.local_data, jac


class DiscParallelLinearization(DiscParallelExecution):
    """Linearize disciplines in parallel."""

    @staticmethod
    def _get_callables(  # noqa:D102
        disciplines: Sequence[MDODiscipline],
    ) -> list[Callable]:
        return [_Functor(d) for d in disciplines]

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
        if len(self._disciplines) == 1 or not len(self._disciplines) == len(
            self.inputs
        ):
            if len(self._disciplines) == 1:
                self.workers[0].local_data = ordered_outputs[0][0]
                self.workers[0].jac = ordered_outputs[0][1]
            if IS_WIN and not self.use_threading:
                disc = self._disciplines[0]
                # Only increase the number of calls if the Jacobian was computed.
                if ordered_outputs[0][0]:
                    disc.n_calls += len(self.inputs)
                    disc.n_calls_linearize += len(self.inputs)
        else:
            for disc, output in zip(self.workers, ordered_outputs):
                # When the discipline in the worker failed, output is None.
                # We do not update the local_data such that the issue is caught by the
                # output grammar.
                if output[0] is not None:
                    disc.local_data = output[0]
                disc.jac = output[1]

        return [out[1] for out in ordered_outputs]
