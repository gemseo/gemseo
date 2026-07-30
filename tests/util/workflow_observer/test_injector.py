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

"""Tests for the workflow observer injector."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

import pytest

from gemseo.util._workflow_observer.base_observer import ObservationSpec
from gemseo.util._workflow_observer.injector import _decorate_class
from gemseo.util._workflow_observer.injector import _match_spec
from gemseo.util._workflow_observer.injector import _WorkflowObserverInjector
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.util._workflow_observer.interface import CallArguments
    from gemseo.util._workflow_observer.interface import CallSpec


class _StubObserver:
    """Minimal observer that records start/end/init calls without using the factory."""

    _spec: ClassVar[ObservationSpec] = ObservationSpec(
        base_class="tests.utils.workflow_observer.test_injector._Target",
        method_names_for_start={"only_start"},
        method_names_for_finish={"only_finish"},
        method_names_for_both={"both"},
    )

    def __init__(self, object_: object, init_arguments: CallArguments) -> None:
        self.events: list[tuple[str, Any]] = []

    def start(self, call_spec: CallSpec) -> None:
        self.events.append(("start", call_spec.callable_.__name__))

    def end(self, call_spec: CallSpec, returned_data: Any) -> None:
        self.events.append(("end", call_spec.callable_.__name__))


class _Target:
    def __init__(self) -> None:
        pass

    def only_start(self, x: int) -> int:
        return x + 1

    def only_finish(self, x: int) -> int:
        if x < 0:
            msg = "boom"
            raise ValueError(msg)
        return x * 2

    def both(self, x: int) -> int:
        return x


@pytest.fixture(scope="module", autouse=True)
def _decorated_target():
    """Decorate `_Target` once before the tests that exercise the wrappers run."""
    _decorate_class(_StubObserver._spec, _StubObserver, _Target)


@pytest.fixture
def enable_observation(monkeypatch):
    """Enable the workflow observation, required for the wrappers to notify."""
    from gemseo.util._directory_manager.settings import Settings
    from gemseo.util.global_configuration import _configuration

    # The manager cannot be disabled once enabled, so swap in a fresh enabled
    # settings instance: monkeypatch restores the previous one on teardown.
    settings = Settings()
    settings.enable = True
    monkeypatch.setattr(_configuration, "directory_manager", settings)


def test_inject_raises_when_no_observer_matches(snapshot):
    class _UnobservedClass:
        pass

    with assert_exception(RuntimeError, snapshot):
        _WorkflowObserverInjector.inject(_UnobservedClass)


def test_decorate_with_start_runs_underlying_method(enable_observation):
    target = _Target()
    assert target.only_start(1) == 2
    assert ("start", "only_start") in target._workflow_observer.events


def test_decorate_with_finish_propagates_exception_and_calls_end(
    enable_observation, snapshot
):
    target = _Target()
    with assert_exception(ValueError, snapshot):
        target.only_finish(-1)
    assert ("end", "only_finish") in target._workflow_observer.events


def test_decorate_with_both_calls_start_and_end(enable_observation):
    target = _Target()
    assert target.both(7) == 7
    events = target._workflow_observer.events
    assert ("start", "both") in events
    assert ("end", "both") in events


def test_no_observation_when_disabled():
    """An object created and called while disabled is executed unobserved."""
    target = _Target()
    assert not hasattr(target, "_workflow_observer")
    assert target.both(7) == 7
    assert target.only_start(1) == 2
    assert target.only_finish(1) == 2


def test_inherited_wrapper_defers_to_subclass_wrapper(enable_observation):
    """A decorated override calling super() notifies the observer only once."""

    class _Child(_Target):
        def only_start(self, x: int) -> int:
            return super().only_start(x) + 10

        def only_finish(self, x: int) -> int:
            return super().only_finish(x) + 10

        def both(self, x: int) -> int:
            return super().both(x) + 10

    _decorate_class(_StubObserver._spec, _StubObserver, _Child)
    child = _Child()

    assert child.only_start(1) == 12
    assert child.only_finish(1) == 12
    assert child.both(1) == 11

    events = child._workflow_observer.events
    assert events.count(("start", "only_start")) == 1
    assert events.count(("end", "only_finish")) == 1
    assert events.count(("start", "both")) == 1
    assert events.count(("end", "both")) == 1


def test_failing_observation_end_does_not_mask_the_exception(
    enable_observation, caplog
):
    """A failing observation end is logged, not raised.

    Otherwise it would mask the exception of the observed method.
    """

    class _FailingEndObserver(_StubObserver):
        def end(self, call_spec: CallSpec, returned_data: Any) -> None:
            msg = "end failure"
            raise RuntimeError(msg)

    class _FailingTarget:
        def __init__(self) -> None:
            pass

        def only_finish(self, x: int) -> int:
            if x < 0:
                msg = "boom"
                raise ValueError(msg)
            return x

    _FailingEndObserver._spec = ObservationSpec(
        base_class=f"{_FailingTarget.__module__}.{_FailingTarget.__qualname__}",
        method_names_for_finish={"only_finish"},
    )
    _decorate_class(_FailingEndObserver._spec, _FailingEndObserver, _FailingTarget)

    # The exception of the observed method propagates unchanged...
    with pytest.raises(ValueError, match="boom"):
        _FailingTarget().only_finish(-1)

    # ... and the failure of the observation end is logged.
    assert "The observation of" in caplog.text
    assert "could not be ended" in caplog.text


def test_match_spec_excludes_subclass():
    class _Base:
        pass

    class _Sub(_Base):
        pass

    spec = ObservationSpec(
        base_class=f"{_Base.__module__}.{_Base.__name__}",
        excluded_sub_classes={f"{_Sub.__module__}.{_Sub.__name__}"},
    )
    assert _match_spec(spec, _Base)
    assert not _match_spec(spec, _Sub)


def test_accept_returns_false_when_directory_manager_disabled(monkeypatch):
    from gemseo.util import global_configuration
    from gemseo.util._directory_manager.settings import Settings

    # A fresh settings instance is disabled by default; swap it in so the
    # global is not flipped (the manager cannot be disabled once enabled).
    monkeypatch.setattr(
        global_configuration._configuration, "directory_manager", Settings()
    )

    class _C:
        pass

    assert not _WorkflowObserverInjector.accept(_C)


def test_accept_returns_false_for_abstract_class(monkeypatch):
    from gemseo.util import global_configuration
    from gemseo.util._directory_manager.settings import Settings

    settings = Settings()
    settings.enable = True
    monkeypatch.setattr(
        global_configuration._configuration, "directory_manager", settings
    )

    class _AbstractC(ABC):  # noqa: B024
        pass

    assert not _WorkflowObserverInjector.accept(_AbstractC)
