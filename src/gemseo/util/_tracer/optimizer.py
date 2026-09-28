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

"""Optimizer tracer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.util._tracer.base import BaseTracer

if TYPE_CHECKING:
    from gemseo.optimization.core.base_optimization_library import (
        BaseOptimizationLibrary,
    )
    from gemseo.util._workflow_observer.optimizer import OptimizerWorkflowObserver
    from gemseo.util.typing import MutableStrKeyMapping
    from gemseo.util.typing import StrKeyMapping


class OptimizerTracer(BaseTracer):
    """Tracer for recording optimization algorithm execution data.

    Records traces for optimizer iterations, including the current iteration
    number read from the bound observer.
    """

    _observed_object: BaseOptimizationLibrary

    __observer: OptimizerWorkflowObserver
    """The workflow observer, which owns the iteration counter."""

    def __init__(  # noqa: D107
        self,
        observer: OptimizerWorkflowObserver,
        init_arguments: StrKeyMapping,
    ) -> None:
        super().__init__(observer, init_arguments)
        # The base tracer only keeps the observee: the observer is also
        # needed here, for the iteration counter read in _get_start_trace.
        self.__observer = observer

    def _get_start_trace(self, call_arguments: StrKeyMapping) -> MutableStrKeyMapping:  # noqa: D102
        trace = super()._get_start_trace(call_arguments)
        iteration = self.__observer.iteration
        if iteration is not None:
            # The observer has no iteration number until the observation of
            # `execute` has captured the evaluation counter of the problem: the
            # iteration is then simply omitted. This is told from the value
            # rather than from an `AttributeError`, which would also hide a
            # genuine error raised while reading the counter.
            trace["iteration"] = iteration
        return trace
