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

"""Tests for directory manager processors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from numpy import array

from gemseo.doe.core.base_doe_library import BaseDOELibrary
from gemseo.util._directory_manager.manager import DirectoryManager
from gemseo.util._directory_manager.processor.discipline import (
    DisciplineExecutionDMProcessor,
)
from gemseo.util._directory_manager.processor.discipline import (
    DisciplineLinearizationDMProcessor,
)
from gemseo.util._directory_manager.processor.doe import DOEDMProcessor
from gemseo.util._directory_manager.processor.mda import MDAExecutionDMProcessor
from gemseo.util._directory_manager.processor.mda import MDAIterationDMProcessor
from gemseo.util._directory_manager.processor.optimizer import OptimizerDMProcessor
from gemseo.util._directory_manager.processor.scenario import ScenarioDMProcessor
from gemseo.util._tracer._yaml import convert_to_yaml_data
from gemseo.util._tracer.scenario import ScenarioTracer
from gemseo.util._workflow_observer.doe import DOEWorkflowObserver
from gemseo.util._workflow_observer.interface import CallSpec
from gemseo.util._workflow_observer.scenario import ScenarioWorkflowObserver

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping


@pytest.fixture
def empty_call_arguments() -> StrKeyMapping:
    return {}


def _make_observer(observer_class, **attrs):
    """Build a bare observer with attributes set, skipping its `__init__`."""
    observer = observer_class.__new__(observer_class)
    for key, value in attrs.items():
        setattr(observer, key, value)
    return observer


class _NamedObject:
    """Object whose `str()` returns a fixed name."""

    def __init__(self, name: str) -> None:
        self.__name = name

    def __str__(self) -> str:
        return self.__name


def test_base_str_returns_observed_object_name():
    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))
    processor = ScenarioDMProcessor.__new__(ScenarioDMProcessor)
    processor._observer = observer
    assert str(processor) == "scen"


def _patch_start_directory(
    monkeypatch, calls: list, directory_path: Path = Path("dir")
) -> None:
    """Replace `start_directory` by a recording stub returning a fixed path.

    Args:
        monkeypatch: The fixture patching `DirectoryManager`.
        calls: The list where the stub records the arguments it is called with.
        directory_path: The path returned by the stub.
    """

    def _start_directory(self, observer, name) -> Path:
        """Record the arguments of the call and return the fixed path.

        Args:
            self: The directory manager the stub is bound to.
            observer: The observer the directory is started for.
            name: The name of the directory.

        Returns:
            The fixed path.
        """
        calls.append((observer, name))
        return directory_path

    monkeypatch.setattr(DirectoryManager, "start_directory", _start_directory)


def test_base_start_delegates_to_directory_manager(
    monkeypatch, tmp_wd, empty_call_arguments
):
    calls: list = []
    _patch_start_directory(monkeypatch, calls)
    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))
    processor = ScenarioDMProcessor(observer, empty_call_arguments)
    # This test only cares about the processor-to-directory-manager
    # delegation, not about the tracer, which is exercised by the tracer
    # package's own tests.
    processor._tracer = SimpleNamespace(start=lambda call_spec, directory_path: None)

    processor.start(CallSpec(kwargs={}, callable_=str))

    assert calls == [(observer, "scen")]


def test_base_end_delegates_to_directory_manager(
    monkeypatch, tmp_wd, empty_call_arguments
):
    calls: list = []
    _patch_start_directory(monkeypatch, [])
    monkeypatch.setattr(
        DirectoryManager,
        "end_directory",
        lambda self, observer: calls.append(observer),
    )
    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))
    processor = ScenarioDMProcessor(observer, empty_call_arguments)
    # This test only cares about the processor-to-directory-manager
    # delegation, not about the tracer, which is exercised by the tracer
    # package's own tests.
    processor._tracer = SimpleNamespace(
        start=lambda call_spec, directory_path: None,
        end=lambda call_spec, returned_data, directory_path: None,
    )

    processor.start(CallSpec(kwargs={}, callable_=str))
    processor.end(CallSpec(kwargs={}, callable_=str), returned_data=None)

    assert calls == [observer]


def test_base_end_traces_in_the_directory_returned_by_start(
    monkeypatch, tmp_wd, empty_call_arguments
):
    """The tracer must write in the directory that `start_directory` returned.

    The manager maps a directory path to an observer, so searching the path
    back from the observer is a linear scan over every directory started since
    the beginning of the process.
    """
    _patch_start_directory(monkeypatch, [], directory_path=Path("started_here"))
    monkeypatch.setattr(DirectoryManager, "end_directory", lambda self, observer: None)
    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))
    processor = ScenarioDMProcessor(observer, empty_call_arguments)
    traced_paths: list[Path] = []
    processor._tracer = SimpleNamespace(
        start=lambda call_spec, directory_path: None,
        end=lambda call_spec, returned_data, directory_path: traced_paths.append(
            directory_path
        ),
    )

    processor.start(CallSpec(kwargs={}, callable_=str))
    processor.end(CallSpec(kwargs={}, callable_=str), returned_data=None)

    assert traced_paths == [Path("started_here")]


def test_base_end_logs_the_error_when_the_cycle_has_not_started(
    monkeypatch, tmp_wd, empty_call_arguments, caplog
):
    """Verify that an unstarted cycle is logged like any other tracing error.

    `BaseWorkflowObserver.end` refuses to end an observation that has not
    started, so this cannot happen through an observer; the directory path is
    nonetheless read inside the guarded block rather than before it.
    """
    calls: list = []
    monkeypatch.setattr(
        DirectoryManager,
        "end_directory",
        lambda self, observer: calls.append(observer),
    )
    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))
    processor = ScenarioDMProcessor(observer, empty_call_arguments)

    processor.end(CallSpec(kwargs={}, callable_=str), returned_data=None)

    assert calls == [observer]
    assert "The tracing of scen could not be ended." in caplog.text
    assert "_BaseDMProcessor__directory_path" in caplog.text


def test_base_start_logs_the_error_when_tracer_start_raises(
    monkeypatch, tmp_wd, empty_call_arguments, caplog
):
    """A tracer error must be logged instead of breaking the observed call."""
    calls: list = []
    _patch_start_directory(monkeypatch, [])
    monkeypatch.setattr(
        DirectoryManager,
        "end_directory",
        lambda self, observer: calls.append(observer),
    )
    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))
    processor = ScenarioDMProcessor(observer, empty_call_arguments)

    class _TracerStartError(RuntimeError):
        """The tracer's simulated failure."""

    def _raise_start(call_arguments, directory_path):
        raise _TracerStartError

    processor._tracer = SimpleNamespace(start=_raise_start)

    processor.start(CallSpec(kwargs={}, callable_=str))

    # The directory is left open: the observed call is about to run in it and
    # end() closes it.
    assert calls == []
    assert "The tracing of scen could not be started." in caplog.text
    assert "_TracerStartError" in caplog.text


def test_base_start_restores_directory_when_tracer_start_is_interrupted(
    monkeypatch, tmp_wd, empty_call_arguments
):
    """An exception that is not an error must propagate, with the directory ended."""
    calls: list = []
    _patch_start_directory(monkeypatch, [])
    monkeypatch.setattr(
        DirectoryManager,
        "end_directory",
        lambda self, observer: calls.append(observer),
    )
    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))
    processor = ScenarioDMProcessor(observer, empty_call_arguments)

    def _raise_start(call_arguments, directory_path):
        raise KeyboardInterrupt

    processor._tracer = SimpleNamespace(start=_raise_start)

    with pytest.raises(KeyboardInterrupt):
        processor.start(CallSpec(kwargs={}, callable_=str))

    # Otherwise the working directory would stay inside the execution directory.
    assert calls == [observer]


def test_base_end_logs_the_error_and_ends_the_directory_when_tracer_end_raises(
    monkeypatch, tmp_wd, empty_call_arguments, caplog
):
    """A tracer error must be logged, with the directory ended nonetheless."""
    calls: list = []
    _patch_start_directory(monkeypatch, [])
    monkeypatch.setattr(
        DirectoryManager,
        "end_directory",
        lambda self, observer: calls.append(observer),
    )
    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))
    processor = ScenarioDMProcessor(observer, empty_call_arguments)

    class _TracerEndError(RuntimeError):
        """The tracer's simulated failure."""

    def _raise_end(call_spec, returned_data, directory_path):
        raise _TracerEndError

    processor._tracer = SimpleNamespace(
        start=lambda call_spec, directory_path: None, end=_raise_end
    )

    processor.start(CallSpec(kwargs={}, callable_=str))
    processor.end(CallSpec(kwargs={}, callable_=str), returned_data=None)

    assert calls == [observer]
    assert "The tracing of scen could not be ended." in caplog.text
    assert "_TracerEndError" in caplog.text


def test_base_init_falls_back_to_a_no_op_tracer_when_tracer_construction_raises(
    tmp_wd, caplog
):
    """Tracer construction must never abort the observed object's constructor."""

    class _RaisingStr:
        """Object whose `str()` raises."""

        def __str__(self) -> str:
            msg = "boom"
            raise RuntimeError(msg)

    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))

    processor = ScenarioDMProcessor(observer, {"x": _RaisingStr()})

    assert (
        "The tracing of an instance of ScenarioDMProcessor could not be constructed."
        in caplog.text
    )
    assert "RuntimeError" in caplog.text
    # The processor is left usable: start/end on the fallback tracer are no-ops
    # rather than raising again.
    processor._tracer.start({}, tmp_wd)
    processor._tracer.end({}, None, tmp_wd)


def test_base_init_traces_a_self_referential_init_argument_as_a_cycle(tmp_wd, caplog):
    """A self-referential init argument must be traced, not abort the constructor.

    `convert_to_yaml_data` cuts the cycle with a marker instead of recursing
    into the `Sequence` until the stack is exhausted, so the observee is
    registered and the real tracer is built, rather than the no-op fallback.
    """
    observer = _make_observer(ScenarioWorkflowObserver, object_=_NamedObject("scen"))
    self_referential_list: list = []
    self_referential_list.append(self_referential_list)

    processor = ScenarioDMProcessor(observer, {"x": self_referential_list})

    assert (
        "The tracing of an instance of ScenarioDMProcessor could not be constructed."
        not in caplog.text
    )
    assert isinstance(processor._tracer, ScenarioTracer)
    assert convert_to_yaml_data({"x": self_referential_list}) == {"x": ["<cycle>"]}


def test_mda_execution_str_uses_object_name():
    processor = MDAExecutionDMProcessor.__new__(MDAExecutionDMProcessor)
    processor._observer = SimpleNamespace(object_=_NamedObject("MDAGauss"))
    assert str(processor) == "MDAGauss"


def test_mda_iteration_str_uses_object_name_and_current_iter():
    class _MDA:
        _current_iter = 4

        def __str__(self) -> str:
            return "MDAJacobi"

    processor = MDAIterationDMProcessor.__new__(MDAIterationDMProcessor)
    processor._observer = SimpleNamespace(object_=_MDA())
    assert str(processor) == "MDAJacobi_iteration_4"


def test_optimizer_str_uses_iteration_plus_one():
    processor = OptimizerDMProcessor.__new__(OptimizerDMProcessor)
    processor._observer = SimpleNamespace(iteration=6)
    assert str(processor) == "Optimizer_iteration_7"


def test_optimizer_str_falls_back_to_the_first_iteration():
    """Verify the name used before the observer has an iteration number.

    The observer has none until the observation of `execute` has captured the
    evaluation counter of the problem; the directories being numbered from 1,
    that is the first one.
    """
    processor = OptimizerDMProcessor.__new__(OptimizerDMProcessor)
    processor._observer = SimpleNamespace(iteration=None)
    assert str(processor) == "Optimizer_iteration_1"


def test_discipline_execution_str_appends_execution_suffix():
    processor = DisciplineExecutionDMProcessor.__new__(DisciplineExecutionDMProcessor)
    processor._observer = SimpleNamespace(object_=_NamedObject("Sellar1"))
    assert str(processor) == "Sellar1_execution"


def test_discipline_linearization_str_appends_linearization_suffix():
    processor = DisciplineLinearizationDMProcessor.__new__(
        DisciplineLinearizationDMProcessor
    )
    processor._observer = SimpleNamespace(object_=_NamedObject("Sellar2"))
    assert str(processor) == "Sellar2_linearization"


def test_doe_str_uses_sample_index(monkeypatch, tmp_wd, empty_call_arguments):
    """The directory is named after the sample index, not the evaluation order."""
    _patch_start_directory(monkeypatch, [])
    samples = array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    observer = _make_observer(
        DOEWorkflowObserver, object_=SimpleNamespace(samples=samples)
    )
    processor = DOEDMProcessor(observer, empty_call_arguments)

    # Whatever the order in which the samples are evaluated, the directory number
    # matches the position of the sample in the DOE, passed by the DOE library.
    processor.start(
        CallSpec.create_safely(
            BaseDOELibrary._evaluate_functions, (samples[2],), {"sample_index": 2}
        )
    )
    assert str(processor) == "DOE_sample_3"
    processor.start(
        CallSpec.create_safely(
            BaseDOELibrary._evaluate_functions, (samples[0],), {"sample_index": 0}
        )
    )
    assert str(processor) == "DOE_sample_1"

    # Without an index (e.g. all the samples evaluated at once), a single
    # directory is used.
    processor.start(
        CallSpec.create_safely(BaseDOELibrary._evaluate_functions, (samples,), {})
    )
    assert str(processor) == "DOE_samples"
