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
"""An observer for DOE algorithms."""

from __future__ import annotations

from typing import Final

from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver
from gemseo.util._workflow_observer.base_observer import ObservationSpec


class DOEWorkflowObserver(BaseWorkflowObserver):
    """Observer for Design of Experiments (DOE) execution lifecycle.

    Monitors the `_evaluate_functions()` method of DOE algorithms to track
    function evaluation start and end. Observes all `BaseDOELibrary` instances.
    """

    _spec: Final[ObservationSpec] = ObservationSpec(
        base_class="gemseo.doe.core.base_doe_library.BaseDOELibrary",
        method_names_for_both={"_evaluate_functions"},
    )
