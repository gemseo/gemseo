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
"""An observer for optimizers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from gemseo.utils._workflow_observers.base_observer import BaseWorkflowObserver
from gemseo.utils._workflow_observers.base_observer import ObservationSpec

if TYPE_CHECKING:
    from gemseo.algos.evaluation_counter import EvaluationCounter
    from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
    from gemseo.utils._workflow_observers.interface import CallArguments
    from gemseo.utils._workflow_observers.interface import CallSpec


class OptimizerWorkflowObserver(BaseWorkflowObserver):
    """Observer for optimization algorithm execution lifecycle.

    Monitors the `execute()` and `_finalize_previous_iteration()` methods of optimizers,
    and the `_get_early_stopping_result()` method for finish events. Tracks the current
    evaluation counter to report algorithm iteration progress.
    Observes all `BaseOptimizationLibrary` instances.
    """

    _spec: Final[ObservationSpec] = ObservationSpec(
        base_class="gemseo.algos.opt.base_optimization_library.BaseOptimizationLibrary",
        method_names_for_both={
            "execute",
            "_finalize_previous_iteration",
        },
        method_names_for_finish={
            "_get_early_stopping_result",
        },
    )

    __evaluation_counter: EvaluationCounter | None
    """The evaluation counter of the optimization problem."""

    _object: BaseOptimizationLibrary

    def __init__(  # noqa: D107
        self,
        object_: BaseOptimizationLibrary,
        init_arguments: CallArguments,
    ) -> None:
        super().__init__(object_, init_arguments)
        self.__evaluation_counter = None

    def start(self, call_spec: CallSpec) -> None:  # noqa: D102
        # TODO: use dispatcher?
        if call_spec.callable_.__name__ == "execute":
            problem = call_spec.args[0]
            self.__evaluation_counter = problem.evaluation_counter
            super().start(call_spec)
        else:
            super().end(call_spec, None)

    @property
    def iteration(self) -> int:
        """The current iteration number of the optimization algorithm."""
        return self.__evaluation_counter.current

    def end(self, call_spec: CallSpec, returned_data: Any) -> None:  # noqa: D102
        if call_spec.callable_.__name__ == "_finalize_previous_iteration":
            super().start(call_spec)
        elif self._status.is_started:
            super().end(call_spec, returned_data)
