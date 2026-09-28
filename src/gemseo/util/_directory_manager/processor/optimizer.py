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
"""Directory manager for an optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.util._directory_manager.processor.base import BaseDMProcessor
from gemseo.util._tracer.optimizer import OptimizerTracer
from gemseo.util._workflow_observer.optimizer import OptimizerWorkflowObserver

if TYPE_CHECKING:
    from gemseo.util._tracer.base import BaseTracer
    from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver


class OptimizerDMProcessor(BaseDMProcessor):
    """Directory manager for optimization algorithm iteration events.

    Creates and manages directories for each iteration of an optimization algorithm,
    with directory names reflecting the optimizer and current iteration number.
    """

    observer_class: ClassVar[type[BaseWorkflowObserver]] = OptimizerWorkflowObserver

    _tracer_class: ClassVar[type[BaseTracer]] = OptimizerTracer

    _observer: OptimizerWorkflowObserver
    """The workflow observer, which owns the iteration counter."""

    def __str__(self) -> str:
        iteration = self._observer.iteration
        # The observer has no iteration number until the observation of
        # `execute` has captured the evaluation counter of the problem; the
        # directories being numbered from 1, that is the first one.
        number = 1 if iteration is None else iteration + 1
        return f"Optimizer_iteration_{number}"
