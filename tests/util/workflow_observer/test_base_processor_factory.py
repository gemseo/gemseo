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

import pytest

from gemseo.util._directory_manager.processor.factory import DM_PROCESSOR_FACTORY
from gemseo.util._directory_manager.processor.mda import MDAExecutionDMProcessor
from gemseo.util._directory_manager.processor.optimizer import OptimizerDMProcessor
from gemseo.util._directory_manager.processor.scenario import ScenarioDMProcessor
from gemseo.util._workflow_observer.interface import CallArguments
from gemseo.util._workflow_observer.mda import MDAExecutionWorkflowObserver
from gemseo.util._workflow_observer.optimizer import OptimizerWorkflowObserver
from gemseo.util._workflow_observer.scenario import ScenarioWorkflowObserver
from gemseo.util.testing.helper import assert_exception


def test_create_raises_for_unknown_observer(snapshot):
    with assert_exception(ValueError, snapshot):
        DM_PROCESSOR_FACTORY.create(object(), CallArguments(args=(), kwargs={}))


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
    processor = DM_PROCESSOR_FACTORY.create(observer, CallArguments(args=(), kwargs={}))
    assert isinstance(processor, expected_processor_class)
