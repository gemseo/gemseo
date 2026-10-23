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
"""Observers for the MDA."""

from __future__ import annotations

from typing import ClassVar
from typing import Final

from gemseo.util._workflow_observer.base_dispatcher import (
    BaseWorkflowObserverDispatcher,
)
from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver
from gemseo.util._workflow_observer.base_observer import ObservationSpec


class MDAExecutionWorkflowObserver(BaseWorkflowObserver):
    """Observer for MDA solver execution lifecycle events.

    Monitors the `execute()` method of MDA solvers
    to track overall execution start and end.
    """


class MDAIterationWorkflowObserver(BaseWorkflowObserver):
    """Observer for individual MDA solver iteration lifecycle events.

    Monitors the `_iterate_once()` method to track individual iteration start and end.
    """


class MDAWorkflowObserver(BaseWorkflowObserverDispatcher):
    """Observer that dispatches to MDA-specific observers.

    Routes observation events to either
    `MDAExecutionWorkflowObserver` or `MDAIterationWorkflowObserver`
    based on the method being called. Observes all `BaseMDASolver` instances.
    """

    _spec: Final[ObservationSpec] = ObservationSpec(
        base_class="gemseo.mda.core.base_solver.BaseMDASolver",
        method_names_for_both={"execute", "_iterate_once"},
    )

    _method_name_to_observer_class: ClassVar[dict[str, type[BaseWorkflowObserver]]] = {
        "execute": MDAExecutionWorkflowObserver,
        "_iterate_once": MDAIterationWorkflowObserver,
    }
