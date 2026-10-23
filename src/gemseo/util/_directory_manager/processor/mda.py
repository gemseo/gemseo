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
"""Directory managers for MDA algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.util._directory_manager.processor.base import BaseDMProcessor
from gemseo.util._tracer.mda import MDAExecutionTracer
from gemseo.util._tracer.mda import MDAIterationTracer
from gemseo.util._workflow_observer.mda import MDAExecutionWorkflowObserver
from gemseo.util._workflow_observer.mda import MDAIterationWorkflowObserver

if TYPE_CHECKING:
    from gemseo.util._tracer.base import BaseTracer
    from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver


class MDAExecutionDMProcessor(BaseDMProcessor):
    """Directory manager for MDA solver execution events.

    Creates and manages directories for the overall execution lifecycle
    of an MDA solver.
    """

    observer_class: ClassVar[type[BaseWorkflowObserver]] = MDAExecutionWorkflowObserver

    _tracer_class: ClassVar[type[BaseTracer]] = MDAExecutionTracer


class MDAIterationDMProcessor(BaseDMProcessor):
    """Directory manager for MDA solver iteration events.

    Creates and manages directories for each iteration within an MDA solver execution,
    with directory names reflecting the solver and iteration counter.
    """

    observer_class: ClassVar[type[BaseWorkflowObserver]] = MDAIterationWorkflowObserver

    _tracer_class: ClassVar[type[BaseTracer]] = MDAIterationTracer

    def __str__(self) -> str:
        object_ = self._observer.object_
        return f"{object_}_iteration_{object_._current_iter}"
