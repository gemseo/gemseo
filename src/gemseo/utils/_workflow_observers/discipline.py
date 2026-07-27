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
"""Observers for disciplines."""

from __future__ import annotations

from typing import ClassVar
from typing import Final

from gemseo.utils._workflow_observers.base_dispatcher import (
    BaseWorkflowObserverDispatcher,
)
from gemseo.utils._workflow_observers.base_observer import BaseWorkflowObserver
from gemseo.utils._workflow_observers.base_observer import ObservationSpec


class DisciplineExecutionWorkflowObserver(BaseWorkflowObserver):
    """Observer for discipline execution lifecycle events.

    Monitors the `execute()` method of disciplines to track execution start and end.
    """


class DisciplineLinearizationWorkflowObserver(BaseWorkflowObserver):
    """Observer for discipline linearization lifecycle events.

    Monitors the `linearize()` method of disciplines
    to track linearization start and end.
    """


class DisciplineWorkflowObserver(BaseWorkflowObserverDispatcher):
    """Observer that dispatches to discipline-specific observers.

    Routes observation events to either `DisciplineExecutionWorkflowObserver` or
    `DisciplineLinearizationWorkflowObserver` based on the method being called.
    Observes all `Discipline` instances
    except `ProcessDiscipline` and `DummyDiscipline`.
    """

    _spec: Final[ObservationSpec] = ObservationSpec(
        base_class="gemseo.core.discipline.discipline.Discipline",
        excluded_sub_classes={
            "gemseo.core.process_discipline.ProcessDiscipline",
            "gemseo.utils.discipline.DummyDiscipline",
        },
        method_names_for_both={"execute", "linearize"},
    )

    _method_name_to_observer_class: ClassVar[dict[str, type[BaseWorkflowObserver]]] = {
        "execute": DisciplineExecutionWorkflowObserver,
        "linearize": DisciplineLinearizationWorkflowObserver,
    }
