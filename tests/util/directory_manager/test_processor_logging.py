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

"""Tests for the logging of the directory manager processors.

The error logged when the construction of a tracer fails is formatted by the
log handlers, `MultiLineHandlerMixin.emit` doing so outside the guard of
`logging.Handler.emit`. Anything read while formatting it must therefore
already exist, otherwise the construction of the observed object would be
aborted by the very handler meant to report the failure.
"""

from __future__ import annotations

import pytest

from gemseo.util._directory_manager.processor.doe import DOEDMProcessor
from gemseo.util._directory_manager.processor.optimizer import OptimizerDMProcessor
from gemseo.util._workflow_observer.doe import DOEWorkflowObserver
from gemseo.util._workflow_observer.optimizer import OptimizerWorkflowObserver


class _RaisingStr:
    """Object whose `str()` raises, making the tracer construction fail."""

    def __str__(self) -> str:
        msg = "boom"
        raise RuntimeError(msg)


class _NamedObject:
    """Object whose `str()` returns a fixed name."""

    def __init__(self, name: str) -> None:
        self.__name = name

    def __str__(self) -> str:
        return self.__name


@pytest.fixture
def raising_init_arguments() -> dict[str, _RaisingStr]:
    """Init arguments whose tracing raises."""
    return {"x": _RaisingStr()}


def test_doe_init_falls_back_to_a_no_op_tracer_when_tracer_construction_raises(
    tmp_wd, caplog, raising_init_arguments
):
    """The DOE processor must survive a tracer construction error.

    Its `__str__` reads the sample index, which `start` has not set yet when
    the error is logged from the constructor.
    """
    observer = DOEWorkflowObserver.__new__(DOEWorkflowObserver)
    observer.object_ = _NamedObject("doe")

    processor = DOEDMProcessor(observer, raising_init_arguments)

    assert (
        "The tracing of an instance of DOEDMProcessor could not be constructed."
        in caplog.text
    )
    assert "RuntimeError" in caplog.text
    # The processor is nonetheless usable: its name is available before `start`
    # and its fallback tracer does nothing instead of raising again.
    assert str(processor) == "DOE_samples"
    processor._tracer.start({}, tmp_wd)
    processor._tracer.end({}, None, tmp_wd)


def test_optimizer_init_falls_back_to_a_no_op_tracer_when_tracer_construction_raises(
    tmp_wd, caplog, raising_init_arguments
):
    """The optimizer processor must survive a tracer construction error.

    The processor is built by the constructor of its observer, whose `__str__`
    reads the iteration number of that very observer.
    """
    observer = OptimizerWorkflowObserver(_NamedObject("opt"), raising_init_arguments)

    assert (
        "The tracing of an instance of OptimizerDMProcessor could not be constructed."
        in caplog.text
    )
    assert "RuntimeError" in caplog.text
    assert observer.iteration is None
    processor = OptimizerDMProcessor.__new__(OptimizerDMProcessor)
    processor._observer = observer
    assert str(processor) == "Optimizer_iteration_1"
