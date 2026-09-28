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

"""Tests for the workflow observer processor factory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gemseo.util._directory_manager.processor.factory import dm_processor_factory
from gemseo.util._directory_manager.processor.mda import MDAExecutionDMProcessor
from gemseo.util._directory_manager.processor.optimizer import OptimizerDMProcessor
from gemseo.util._directory_manager.processor.scenario import ScenarioDMProcessor
from gemseo.util._workflow_observer.mda import MDAExecutionWorkflowObserver
from gemseo.util._workflow_observer.optimizer import OptimizerWorkflowObserver
from gemseo.util._workflow_observer.scenario import ScenarioWorkflowObserver
from gemseo.util.testing.helper import assert_exception


def test_create_raises_for_unknown_observer(snapshot):
    with assert_exception(ValueError, snapshot):
        dm_processor_factory.create(object(), {})


@pytest.mark.parametrize(
    ("observer_class", "expected_processor_class"),
    [
        (ScenarioWorkflowObserver, ScenarioDMProcessor),
        (OptimizerWorkflowObserver, OptimizerDMProcessor),
        (MDAExecutionWorkflowObserver, MDAExecutionDMProcessor),
    ],
)
def test_create_returns_processor_matching_observer_type(
    tmp_wd,  # noqa: ARG001  # Isolate cwd from DirectoryManager singleton chdir.
    observer_class,
    expected_processor_class,
):
    observer = observer_class.__new__(observer_class)
    # The tracer created by the processor needs an observed object to
    # register in the trace registry (class name, name, documentation); in
    # real use this is always set by `BaseWorkflowObserver.__init__` before
    # the processor is created. `SimpleNamespace` (rather than `object()`) can
    # carry the `_workflow_trace_id` the tracer stamps back onto it.
    observer.object_ = SimpleNamespace()
    processor = dm_processor_factory.create(observer, {})
    assert isinstance(processor, expected_processor_class)
