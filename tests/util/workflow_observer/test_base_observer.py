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

"""Tests for the base workflow observer."""

from __future__ import annotations

from gemseo.util._workflow_observer.base_observer import Status
from gemseo.util._workflow_observer.interface import CallSpec
from gemseo.util._workflow_observer.scenario import ScenarioWorkflowObserver
from gemseo.util.testing.helper import assert_exception


def _make_call_spec() -> CallSpec:
    return CallSpec(callable_=lambda: None, args=(), kwargs={})


def test_start_raises_when_already_started(snapshot):
    observer = ScenarioWorkflowObserver.__new__(ScenarioWorkflowObserver)
    observer._status = Status(is_started=True)
    with assert_exception(RuntimeError, snapshot):
        observer.start(_make_call_spec())


def test_end_raises_when_not_started(snapshot):
    observer = ScenarioWorkflowObserver.__new__(ScenarioWorkflowObserver)
    observer._status = Status(is_started=False)
    with assert_exception(RuntimeError, snapshot):
        observer.end(_make_call_spec(), None)
