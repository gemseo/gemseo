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

from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver
from gemseo.util._workflow_observer.base_observer import ObservationSpec

if TYPE_CHECKING:
    from gemseo.core.problem.counter import EvaluationCounter
    from gemseo.optimization.core.base_optimization_library import (
        BaseOptimizationLibrary,
    )
    from gemseo.util._workflow_observer.interface import CallSpec
    from gemseo.util.typing import StrKeyMapping


class OptimizerWorkflowObserver(BaseWorkflowObserver):
    """Observer for optimization algorithm execution lifecycle.

    Monitors the `execute()` and `_finalize_previous_iteration()` methods of optimizers,
    and the `_get_early_stopping_result()` method for finish events. Tracks the current
    evaluation counter to report algorithm iteration progress.
    Observes all `BaseOptimizationLibrary` instances.
    """

    _spec: Final[ObservationSpec] = ObservationSpec(
        base_class="gemseo.optimization.core.base_optimization_library"
        ".BaseOptimizationLibrary",
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

    object_: BaseOptimizationLibrary

    def __init__(  # noqa: D107
        self,
        object_: BaseOptimizationLibrary,
        init_arguments: StrKeyMapping,
    ) -> None:
        # The counter is set before the base constructor, which builds the
        # processor of this observer: the `__str__` of that processor reads
        # `iteration`, hence this counter, e.g. when the construction of the
        # tracer fails and the error is logged.
        self.__evaluation_counter = None
        super().__init__(object_, init_arguments)

    # Although the events are routed by method name below, this observer is
    # not a candidate for `BaseWorkflowObserverDispatcher`: the dispatcher
    # delegates each method to an independent child observer with its own
    # lifecycle, while here the observed methods drive one shared lifecycle:
    # `_finalize_previous_iteration` closes the observation of the current
    # iteration when it starts and opens the next one when it ends, sharing
    # the status and the evaluation counter captured by `execute`.

    @staticmethod
    def __get_evaluation_counter(
        call_arguments: StrKeyMapping,
    ) -> EvaluationCounter | None:
        """Return the evaluation counter of the problem passed to `execute`.

        The read is defensive: `start` is called outside any `try`, hence an
        error raised here would break the observed call. It is also why the
        `problem` argument is read with a default rather than indexed: it may
        simply be absent from `call_arguments`, e.g. a truly positional
        `execute(self, *args)` has no parameter named `problem` at all, its
        value ending up under the `*args` parameter's own name instead; or a
        subclass may rename the parameter altogether.

        Args:
            call_arguments: The arguments of the observed call, normalized
                by parameter name.

        Returns:
            The evaluation counter of the optimization problem, or `None`
            when the arguments carry no argument holding one.
        """
        return getattr(call_arguments.get("problem"), "evaluation_counter", None)

    def start(self, call_spec: CallSpec) -> None:  # noqa: D102
        if call_spec.callable_.__name__ == "execute":
            # The iteration number is left out of the trace when the evaluation
            # counter cannot be found, instead of breaking the observed call.
            evaluation_counter = self.__get_evaluation_counter(call_spec.kwargs)
            if evaluation_counter is not None:
                self.__evaluation_counter = evaluation_counter
            super().start(call_spec)
        elif self._status.is_started:
            # Symmetric to the guard of `end`: `_finalize_previous_iteration` is
            # called from within `execute`, so an observation is expected to be
            # started when it fires. Were it not, ending an observation that has
            # not started would raise, hence break the observed call, instead of
            # leaving the observation alone.
            super().end(call_spec, None)

    @property
    def iteration(self) -> int | None:
        """The current iteration number of the optimization algorithm.

        `None` until the observation of `execute` has captured the evaluation
        counter of the problem.
        """
        evaluation_counter = self.__evaluation_counter
        if evaluation_counter is None:
            return None
        return evaluation_counter.current

    def end(self, call_spec: CallSpec, returned_data: Any) -> None:  # noqa: D102
        if call_spec.callable_.__name__ == "_finalize_previous_iteration":
            super().start(call_spec)
        elif self._status.is_started:
            super().end(call_spec, returned_data)
