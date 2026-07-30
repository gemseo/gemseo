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

from types import SimpleNamespace

import pytest
from numpy import array

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
from gemseo.util._workflow_observer.doe import DOEWorkflowObserver
from gemseo.util._workflow_observer.interface import CallArguments
from gemseo.util._workflow_observer.interface import CallSpec
from gemseo.util._workflow_observer.scenario import ScenarioWorkflowObserver


@pytest.fixture
def empty_call_arguments() -> CallArguments:
    return CallArguments(args=(), kwargs={})


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
    observer = _make_observer(ScenarioWorkflowObserver, _object=_NamedObject("scen"))
    processor = ScenarioDMProcessor.__new__(ScenarioDMProcessor)
    processor._observer = observer
    assert str(processor) == "scen"


def test_base_start_delegates_to_directory_manager(
    monkeypatch, tmp_wd, empty_call_arguments
):
    calls: list = []
    monkeypatch.setattr(
        DirectoryManager,
        "start_directory",
        lambda self, observer, name: calls.append((observer, name)),
    )
    observer = _make_observer(ScenarioWorkflowObserver, _object=_NamedObject("scen"))
    processor = ScenarioDMProcessor(observer, empty_call_arguments)

    processor.start(call_spec=None)

    assert calls == [(observer, "scen")]


def test_base_end_delegates_to_directory_manager(
    monkeypatch, tmp_wd, empty_call_arguments
):
    calls: list = []
    monkeypatch.setattr(
        DirectoryManager,
        "end_directory",
        lambda self, observer: calls.append(observer),
    )
    observer = _make_observer(ScenarioWorkflowObserver, _object=_NamedObject("scen"))
    processor = ScenarioDMProcessor(observer, empty_call_arguments)

    processor.end(call_spec=None, returned_data=None)

    assert calls == [observer]


def test_mda_execution_str_uses_object_name():
    processor = MDAExecutionDMProcessor.__new__(MDAExecutionDMProcessor)
    processor._observer = SimpleNamespace(_object=_NamedObject("MDAGauss"))
    assert str(processor) == "MDAGauss"


def test_mda_iteration_str_uses_object_name_and_current_iter():
    class _MDA:
        _current_iter = 4

        def __str__(self) -> str:
            return "MDAJacobi"

    processor = MDAIterationDMProcessor.__new__(MDAIterationDMProcessor)
    processor._observer = SimpleNamespace(_object=_MDA())
    assert str(processor) == "MDAJacobi_iteration_4"


def test_optimizer_str_uses_iteration_plus_one():
    processor = OptimizerDMProcessor.__new__(OptimizerDMProcessor)
    processor._observer = SimpleNamespace(iteration=6)
    assert str(processor) == "Optimizer_iteration_7"


def test_discipline_execution_str_appends_execution_suffix():
    processor = DisciplineExecutionDMProcessor.__new__(DisciplineExecutionDMProcessor)
    processor._observer = SimpleNamespace(_object=_NamedObject("Sellar1"))
    assert str(processor) == "Sellar1_execution"


def test_discipline_linearization_str_appends_linearization_suffix():
    processor = DisciplineLinearizationDMProcessor.__new__(
        DisciplineLinearizationDMProcessor
    )
    processor._observer = SimpleNamespace(_object=_NamedObject("Sellar2"))
    assert str(processor) == "Sellar2_linearization"


def test_doe_str_uses_sample_index(monkeypatch, tmp_wd, empty_call_arguments):
    """The directory is named after the sample index, not the evaluation order."""
    monkeypatch.setattr(
        DirectoryManager, "start_directory", lambda self, observer, name: None
    )
    samples = array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    observer = _make_observer(
        DOEWorkflowObserver, _object=SimpleNamespace(samples=samples)
    )
    processor = DOEDMProcessor(observer, empty_call_arguments)

    # Whatever the order in which the samples are evaluated, the directory number
    # matches the position of the sample in the DOE, passed by the DOE library.
    processor.start(
        CallSpec(args=(samples[2],), kwargs={"sample_index": 2}, callable_=str)
    )
    assert str(processor) == "DOE_sample_3"
    processor.start(
        CallSpec(args=(samples[0],), kwargs={"sample_index": 0}, callable_=str)
    )
    assert str(processor) == "DOE_sample_1"

    # Without an index (e.g. all the samples evaluated at once), a single
    # directory is used.
    processor.start(CallSpec(args=(samples,), kwargs={}, callable_=str))
    assert str(processor) == "DOE_samples"
