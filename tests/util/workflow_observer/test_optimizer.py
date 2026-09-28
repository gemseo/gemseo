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

"""Tests for the optimizer workflow observer."""

from __future__ import annotations

from types import SimpleNamespace

from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver
from gemseo.util._workflow_observer.base_observer import Status
from gemseo.util._workflow_observer.interface import CallSpec
from gemseo.util._workflow_observer.optimizer import OptimizerWorkflowObserver


def _finalize_previous_iteration() -> None:
    """Stand for the observed method of the same name."""


def _make_observer(is_started: bool) -> OptimizerWorkflowObserver:
    """Return an observer with a given status and no evaluation counter.

    Args:
        is_started: Whether the observation has started.

    Returns:
        The observer.
    """
    observer = OptimizerWorkflowObserver.__new__(OptimizerWorkflowObserver)
    observer._status = Status(is_started=is_started)
    observer._OptimizerWorkflowObserver__evaluation_counter = None
    return observer


def test_start_leaves_alone_an_observation_that_has_not_started():
    """Verify that closing an observation that has not started does not raise.

    The start event of `_finalize_previous_iteration` closes the observation of
    the current iteration; the method is called from within `execute`, so an
    observation is expected to be started when it fires. Were it not, ending an
    observation that has not started would raise, hence break the observed call.
    """
    observer = _make_observer(is_started=False)

    observer.start(CallSpec(kwargs={}, callable_=_finalize_previous_iteration))

    assert not observer._status.is_started


def test_iteration_is_none_before_the_evaluation_counter_is_captured():
    """Verify that the iteration is `None` while there is no evaluation counter.

    The counter is captured by the observation of `execute`; the tracer and the
    directory manager processor tell that case from this value.
    """
    assert _make_observer(is_started=False).iteration is None


def test_iteration_reads_the_captured_evaluation_counter():
    """Verify that the iteration is read from the captured evaluation counter."""
    observer = _make_observer(is_started=False)
    observer._OptimizerWorkflowObserver__evaluation_counter = SimpleNamespace(current=3)

    assert observer.iteration == 3


def test_start_leaves_the_iteration_unknown_without_a_problem_argument(monkeypatch):
    """Verify that call arguments without a `problem` argument do not break it.

    The `problem` argument is not guaranteed to be in the call arguments, e.g.
    a variadic `execute` may bind it under a different name, or a subclass may
    rename it; the iteration number is then simply unknown.
    """

    def execute() -> None:
        """Stand for the observed method of the same name."""

    observer = _make_observer(is_started=False)
    # The base observation needs a processor and the observer tree; only the
    # capture of the evaluation counter is under test here.
    monkeypatch.setattr(BaseWorkflowObserver, "start", lambda self, call_spec: None)

    observer.start(CallSpec(kwargs={}, callable_=execute))

    assert observer.iteration is None


def test_start_captures_the_evaluation_counter_of_the_problem_argument(monkeypatch):
    """Verify that the counter is read from the `problem` argument when bound."""

    def execute() -> None:
        """Stand for the observed method of the same name."""

    observer = _make_observer(is_started=False)
    monkeypatch.setattr(BaseWorkflowObserver, "start", lambda self, call_spec: None)
    problem = SimpleNamespace(evaluation_counter=SimpleNamespace(current=7))

    observer.start(CallSpec(kwargs={"problem": problem}, callable_=execute))

    assert observer.iteration == 7


def test_start_ignores_a_problem_argument_without_an_evaluation_counter(monkeypatch):
    """Verify that a `problem` without an evaluation counter cannot break the call.

    The call arguments say nothing about what the `problem` argument holds.
    Reading its evaluation counter unguarded would raise out of the observer
    wrapper, which calls `start` outside any `try`, hence abort the observed
    optimization run; the iteration number is left unknown instead.
    """

    def execute() -> None:
        """Stand for the observed method of the same name."""

    observer = _make_observer(is_started=False)
    monkeypatch.setattr(BaseWorkflowObserver, "start", lambda self, call_spec: None)

    observer.start(CallSpec(kwargs={"problem": "not a problem"}, callable_=execute))

    assert observer.iteration is None
