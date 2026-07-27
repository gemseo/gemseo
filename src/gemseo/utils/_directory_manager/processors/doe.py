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
"""Directory managers for DOE algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.utils._directory_manager.processors.base import BaseDMProcessor
from gemseo.utils._workflow_observers.doe import DOEWorkflowObserver

if TYPE_CHECKING:
    from gemseo.utils._workflow_observers.base_observer import BaseWorkflowObserver
    from gemseo.utils._workflow_observers.interface import CallSpec


class DOEDMProcessor(BaseDMProcessor):
    """Directory manager for DOE algorithm sample evaluation events.

    Creates and manages a directory for each sample evaluation in a DOE. The
    directory is named after the index of the sample in the DOE, so that the name
    is reproducible whatever the order in which the samples are evaluated. A DOE
    can be executed in parallel, in which case the samples are evaluated
    concurrently and in a nondeterministic order; a counter would then give the
    same sample a different number from one run to the next. The sample index also
    matches the database iteration used by the clean-up policies to keep the
    baseline and solution directories. The index is passed by the DOE library to
    the observed `_evaluate_functions` method, so that duplicated samples get
    distinct directories.
    """

    observer_class: ClassVar[type[BaseWorkflowObserver]] = DOEWorkflowObserver

    __sample_index: int
    """The zero-based index of the current sample.

    A negative value means that all the samples are evaluated at once.
    """

    def start(self, call_spec: CallSpec) -> None:  # noqa: D102
        self.__sample_index = call_spec.kwargs.get("sample_index", -1)
        super().start(call_spec)

    def __str__(self) -> str:
        if self.__sample_index < 0:
            # All the samples are evaluated in a single call.
            return "DOE_samples"
        return f"DOE_sample_{self.__sample_index + 1}"
