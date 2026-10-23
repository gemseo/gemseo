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

"""Tests for the directory manager tracers."""

from __future__ import annotations

import re
from collections import UserString
from datetime import date
from datetime import datetime
from pathlib import Path
from time import sleep
from types import MappingProxyType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

import pytest
import yaml
from numpy import arange
from numpy import array
from numpy import array_equal
from numpy import bytes_
from numpy import datetime64
from numpy import float64
from numpy import load as load_npy
from numpy import longdouble
from numpy import str_
from pandas import Timestamp
from pydantic import BaseModel

from gemseo import create_scenario
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.doe.core.base_doe_library import BaseDOELibrary
from gemseo.doe.scipy.settings.lhs import LHS_Settings
from gemseo.formulation.mdf_settings import MDF_Settings
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.space.design import DesignSpace
from gemseo.util._directory_manager.settings import Settings
from gemseo.util._tracer import _yaml
from gemseo.util._tracer._yaml import _max_inline_array_size
from gemseo.util._tracer._yaml import load_yaml_trace
from gemseo.util._tracer.base import BaseTracer
from gemseo.util._tracer.discipline import DisciplineExecutionTracer
from gemseo.util._tracer.discipline import DisciplineLinearizationTracer
from gemseo.util._tracer.doe import DOETracer
from gemseo.util._tracer.mda import MDAExecutionTracer
from gemseo.util._tracer.mda import MDAIterationTracer
from gemseo.util._tracer.optimizer import OptimizerTracer
from gemseo.util._tracer.registry import TraceRegistry
from gemseo.util._tracer.scenario import ScenarioTracer
from gemseo.util._workflow_observer.interface import normalize_arguments_safely
from gemseo.util.base_multiton import BaseMultiton
from gemseo.util.base_name_generator import BaseNameGenerator
from gemseo.util.discipline import DummyDiscipline
from gemseo.util.global_configuration import _configuration

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.util.typing import MutableStrKeyMapping
    from gemseo.util.typing import StrKeyMapping


@pytest.fixture(autouse=True)
def _reset_trace_registry():
    """Ensure that every test starts from, and leaves, an empty registry.

    Without this, a root path set by an earlier test (directly, or as a
    side effect of a real processor construction under `dm_settings`) could
    leak into a later one that builds a tracer directly, e.g. writing a
    registry entry under a directory the earlier test owned.
    """
    BaseMultiton.clear_cache(TraceRegistry)
    yield
    BaseMultiton.clear_cache(TraceRegistry)


class _DocumentedObservee:
    """A documented fake observee used to check the tracer's init trace."""


_no_init_arguments: Final[StrKeyMapping] = MappingProxyType({})
"""No constructor arguments, when the tracer under test does not read them."""

_large_size: Final[int] = _max_inline_array_size + 1
"""The smallest number of elements of an array written to a `.npy` file."""


def _load_trace(directory: Path) -> StrKeyMapping:
    """Return the trace written in a directory.

    Args:
        directory: The directory holding the trace file.

    Returns:
        The data of the trace file.
    """
    return yaml.safe_load((directory / ".gemseo-trace.yml").read_text())


class _RecordingTracer(BaseTracer):
    """A minimal concrete tracer exercising `BaseTracer`'s shared behavior.

    `_get_start_trace` and `_get_end_trace` return whatever extra mapping is
    passed through the `start_trace` / `end_trace` call arguments, so each
    test controls exactly what ends up in the trace.
    """

    def _get_start_trace(self, call_arguments: StrKeyMapping) -> MutableStrKeyMapping:
        """Return the extra `start_trace` mapping, if any.

        Args:
            call_arguments: The normalized arguments of the observed call, by
                parameter name.

        Returns:
            The `start_trace` mapping of the call arguments, empty if absent.
        """
        return dict(call_arguments.get("start_trace", {}))

    def _get_end_trace(
        self, call_arguments: StrKeyMapping, returned_data: Any
    ) -> MutableStrKeyMapping:
        """Return the timing data merged with the extra `end_trace` mapping.

        Args:
            call_arguments: The normalized arguments of the observed call, by
                parameter name.
            returned_data: The data returned by the observed callable.

        Returns:
            The base timing data updated with the `end_trace` mapping of the
            call arguments, if any.
        """
        trace = super()._get_end_trace(call_arguments, returned_data)
        trace.update(call_arguments.get("end_trace", {}))
        return trace


@pytest.fixture
def tracer() -> _RecordingTracer:
    """Return a recording tracer observing a documented fake observee.

    Returns:
        A recording tracer reading no constructor argument.
    """
    return _RecordingTracer(
        SimpleNamespace(object_=_DocumentedObservee()), _no_init_arguments
    )


@pytest.fixture
def convert_payload(tracer, tmp_path) -> Callable[[Any], Any]:
    """Return a function tracing a payload and reading it back from the trace.

    Args:
        tracer: The tracer writing the payload.
        tmp_path: The directory of the trace file.

    Returns:
        A function returning the YAML conversion of a payload.
    """

    def convert(payload: Any) -> Any:
        """Return the YAML conversion of a payload traced by `end`.

        Args:
            payload: The data to be traced.

        Returns:
            The payload read back from the trace file.
        """
        call_arguments = {"end_trace": {"payload": payload}}
        tracer.start(call_arguments, tmp_path)
        tracer.end(call_arguments, None, tmp_path)
        return _load_trace(tmp_path)["payload"]

    return convert


class _Stringifiable:
    """An arbitrary object with no special YAML handling, falling back to `str`."""

    def __str__(self) -> str:
        return "stringified!"


class _PlainModel(BaseModel):
    """A pydantic model which is not settings, hence configures no class."""

    value: float = 1.0


class _CustomStr(UserString):
    """A string-like object which is a `Sequence` of `UserString` items."""


class _CustomInt(int):
    """An integer-like object which PyYAML has no representer for."""


def test_end_writes_object_id_and_type_from_init_trace(tracer, tmp_path):
    """Verify that the written trace carries the `object_id` and `type` of init."""
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["type"] == "_DocumentedObservee"
    # Match the id's shape without pinning its exact index, which depends on
    # how many `_DocumentedObservee` instances this test happens to run after.
    assert re.fullmatch(r"DocumentedObservee/\d+", trace["object_id"])


def test_resolve_object_id_re_registers_a_stale_id_after_a_registry_reset():
    """Verify that an id stamped by a superseded registry is not reused.

    `Settings.__reset_directory_manager` replaces the `TraceRegistry`
    singleton whenever the directory manager is (re-)enabled, restarting its
    per-class counters from 0. An observee whose id was stamped by the
    registry that existed before such a reset must not keep it: reusing it
    would collide with the id the new registry assigns to a distinct
    observee of the same class, both resolving to
    ``"DocumentedObservee/0"``.
    """
    first_observee = _DocumentedObservee()
    _RecordingTracer(SimpleNamespace(object_=first_observee), _no_init_arguments)
    assert first_observee._workflow_trace_id == "DocumentedObservee/0"

    # Simulate the reset performed by `Settings.__reset_directory_manager`.
    BaseMultiton.clear_cache(TraceRegistry)

    second_observee = _DocumentedObservee()
    _RecordingTracer(SimpleNamespace(object_=second_observee), _no_init_arguments)
    # A tracer built for the first observee after the reset.
    _RecordingTracer(SimpleNamespace(object_=first_observee), _no_init_arguments)

    # The second observee, registered first in the new registry, gets its
    # counter's first id.
    assert second_observee._workflow_trace_id == "DocumentedObservee/0"
    # The first observee's stale id is not reused: it is re-registered in the
    # new registry instead of colliding with the second observee's id.
    assert first_observee._workflow_trace_id == "DocumentedObservee/1"


def test_resolve_object_id_reuses_an_id_unpickled_from_another_process(monkeypatch):
    """Verify that an id stamped in another process is never treated as stale.

    A worker process either shares the parent's registry (forked, inheriting
    a snapshot of it) or rebuilds an unrelated one from scratch (a non-fork
    start method); either way, comparing generations across that boundary
    would be meaningless, and the id unpickled from the parent must still be
    reused as is, matching the registry's own handling of that case (see
    `TraceRegistry`'s class documentation).
    """
    observee = _DocumentedObservee()
    _RecordingTracer(SimpleNamespace(object_=observee), _no_init_arguments)
    assert observee._workflow_trace_id == "DocumentedObservee/0"

    # Resetting the registry between the two processes would make the id
    # look stale if the check did not special-case a change of process.
    BaseMultiton.clear_cache(TraceRegistry)
    # Simulate unpickling the observee in a worker process.
    monkeypatch.setattr(
        observee, "_workflow_trace_pid", observee._workflow_trace_pid + 1
    )

    _RecordingTracer(SimpleNamespace(object_=observee), _no_init_arguments)

    assert observee._workflow_trace_id == "DocumentedObservee/0"


def test_registering_writes_documentation_and_init_arguments_to_the_registry(
    dm_settings,
):
    """Verify that the registry entry carries `documentation` and `init_arguments`.

    They are written once, to the registry entry pointed at by `object_id`.
    """
    trace_directory = dm_settings.execution_root_path / "call"
    trace_directory.mkdir()
    # This test builds the tracer directly, bypassing `BaseDMProcessor`,
    # which is what sets the root path in real usage; set it here too,
    # so that registering the observee below actually writes an entry.
    TraceRegistry().set_root_path(dm_settings.execution_root_path)
    tracer = _RecordingTracer(
        SimpleNamespace(object_=_DocumentedObservee()),
        {"x": 1, "y": "b"},
    )
    call_arguments = {}

    tracer.start(call_arguments, trace_directory)
    tracer.end(call_arguments, None, trace_directory)

    trace = _load_trace(trace_directory)
    registry_path = (
        dm_settings.execution_root_path
        / ".gemseo-traces"
        / f"{trace['object_id']}.trace.yml"
    )
    registry_entry = yaml.safe_load(registry_path.read_text())
    assert registry_entry["object_id"] == trace["object_id"]
    assert registry_entry["type"] == "_DocumentedObservee"
    assert registry_entry["documentation"] == _DocumentedObservee.__doc__
    # The normalized constructor arguments are traced by parameter name.
    assert registry_entry["init_arguments"] == {"x": 1, "y": "b"}


def test_end_adds_start_and_duration(tracer, tmp_path):
    """Verify that `start` then `end` add a `start` timestamp and a `duration`."""
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert isinstance(trace["start"], datetime)
    assert isinstance(trace["duration"], float)
    assert trace["duration"] >= 0.0


def test_start_builds_the_trace_before_entering_the_timer(tmp_path):
    """Verify that building the start trace is excluded from the traced duration.

    The start trace, e.g. converting large input arrays to lists, must be
    built before the timer is entered: otherwise that conversion cost would
    be counted in the traced `duration`, and a fast observee call with large
    inputs would get a duration dominated by tracing overhead instead of
    reflecting the observed call itself.
    """

    class _SlowStartTracer(_RecordingTracer):
        """A tracer whose start trace building takes a noticeable time."""

        def _get_start_trace(
            self, call_arguments: StrKeyMapping
        ) -> MutableStrKeyMapping:
            """Sleep before returning, to simulate an expensive trace build.

            Args:
                call_arguments: The normalized arguments of the observed call,
                    by parameter name.

            Returns:
                The start trace of the base recording tracer.
            """
            sleep(0.2)
            return super()._get_start_trace(call_arguments)

    tracer = _SlowStartTracer(
        SimpleNamespace(object_=_DocumentedObservee()), _no_init_arguments
    )
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["duration"] < 0.2


def test_end_omits_start_and_duration_when_start_raised_before_entering_the_timer(
    tmp_path,
):
    """Verify that `end` leaves no garbage timing after `start` failed early.

    `BaseDMProcessor.start` logs and swallows an exception raised while
    building the start trace, e.g. `_get_start_trace` raising because a call
    argument has a raising `__str__` or is a self-referential container, then
    lets the observed call proceed untraced; `end` is still called
    afterward. Since the timer was never entered in that case, `end` must
    not exit it either, and the written trace must have no `start`/
    `duration` keys, instead of a duration computed from the timer's
    construction, e.g. the process' whole uptime.
    """

    class _RaisingOnceStartTracer(_RecordingTracer):
        """A tracer whose first start trace build raises, the next ones don't."""

        _raise_on_start: bool = True

        def _get_start_trace(
            self, call_arguments: StrKeyMapping
        ) -> MutableStrKeyMapping:
            """Raise once, mimicking a call argument with a raising `__str__`.

            Args:
                call_arguments: The normalized arguments of the observed call,
                    by parameter name.

            Returns:
                The start trace of the base recording tracer, from the second
                call on.

            Raises:
                ValueError: At the first call.
            """
            if self._raise_on_start:
                self._raise_on_start = False
                msg = "boom"
                raise ValueError(msg)
            return super()._get_start_trace(call_arguments)

    tracer = _RaisingOnceStartTracer(
        SimpleNamespace(object_=_DocumentedObservee()), _no_init_arguments
    )
    call_arguments = {}
    first_directory = tmp_path / "first"
    first_directory.mkdir()

    with pytest.raises(ValueError, match="boom"):
        tracer.start(call_arguments, first_directory)
    # Mimics `BaseDMProcessor.end`, called even though `start` failed.
    tracer.end(call_arguments, None, first_directory)

    first_trace = _load_trace(first_directory)
    assert "start" not in first_trace
    assert "duration" not in first_trace

    # A later, successful cycle records sane timing again: the failure of the
    # first one must not leave the tracer stuck skipping the timer forever.
    second_directory = tmp_path / "second"
    second_directory.mkdir()
    tracer.start(call_arguments, second_directory)
    tracer.end(call_arguments, None, second_directory)

    second_trace = _load_trace(second_directory)
    assert isinstance(second_trace["start"], datetime)
    assert isinstance(second_trace["duration"], float)
    assert second_trace["duration"] >= 0.0


def test_end_resets_trace_so_a_second_cycle_does_not_accumulate(tracer, tmp_path):
    """Verify that a second start/end cycle does not carry over the first one's data."""
    first_directory = tmp_path / "first"
    first_directory.mkdir()
    second_directory = tmp_path / "second"
    second_directory.mkdir()

    first_call_spec = {"start_trace": {"marker": "first-cycle-only"}}
    tracer.start(first_call_spec, first_directory)
    tracer.end(first_call_spec, None, first_directory)

    second_call_spec = {}
    tracer.start(second_call_spec, second_directory)
    tracer.end(second_call_spec, None, second_directory)

    first_trace = _load_trace(first_directory)
    second_trace = _load_trace(second_directory)
    assert first_trace["marker"] == "first-cycle-only"
    assert "marker" not in second_trace


def test_start_trace_is_converted_when_it_is_captured(tracer, tmp_path):
    """Verify that the start trace holds the values captured at the start.

    A traced value is held by reference, so an observed call mutating one of
    its arguments in place would be traced with the value that argument had
    after the call if the conversion happened only when the trace is written.
    """
    mutated_value = array([1.0])
    call_arguments = {"start_trace": {"value": mutated_value}}

    tracer.start(call_arguments, tmp_path)
    mutated_value += 100.0
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["value"] == [1.0]


def test_yaml_conversion_handles_special_value_types(convert_payload):
    """Verify the recursive YAML conversion of exotic values in an end trace.

    Numpy arrays become lists, numpy scalars become Python scalars, `Path`
    becomes `str`, mappings/sequences/tuples/sets are converted recursively,
    and an arbitrary object with no special handling falls back to `str`.
    """
    payload = {
        "array": array([1.0, 2.0]),
        "scalar": float64(3.5),
        "path": Path("foo") / "bar",
        "nested_mapping": {"inner_array": array([1, 2])},
        "nested_sequence": [array([1]), "text"],
        "a_tuple": (1, 2),
        "a_set": {42},
        "a_frozen_set": frozenset({7}),
        "arbitrary_object": _Stringifiable(),
        "an_enum": BaseNameGenerator.Naming.NUMBERED,
    }
    converted = convert_payload(payload)
    assert converted["array"] == [1.0, 2.0]
    assert converted["scalar"] == 3.5
    assert isinstance(converted["scalar"], float)
    assert converted["path"] == str(Path("foo") / "bar")
    assert converted["nested_mapping"] == {"inner_array": [1, 2]}
    assert converted["nested_sequence"] == [[1], "text"]
    assert converted["a_tuple"] == [1, 2]
    assert converted["a_set"] == [42]
    # A `frozenset` is not a `set`, hence the `AbstractSet` of the conversion:
    # the `str` fallback would give "frozenset({7})".
    assert converted["a_frozen_set"] == [7]
    assert converted["arbitrary_object"] == "stringified!"
    # An enum member, e.g. a default value of a constructor parameter, is
    # converted to its value: PyYAML has no representer for the member itself.
    assert converted["an_enum"] == BaseNameGenerator.Naming.NUMBERED.value


def test_yaml_conversion_handles_non_string_mapping_keys(convert_payload):
    """Verify that a mapping key is converted like a value.

    PyYAML has no more a representer for a `Path`, an enum member or a NumPy
    scalar used as a key than used as a value, so an unconverted key would
    make the dump raise, hence break the observed call.
    """
    payload = {
        Path("foo") / "bar": "path key",
        BaseNameGenerator.Naming.NUMBERED: "enum key",
        float64(1.5): "numpy scalar key",
        (1, 2): "tuple key",
    }
    converted = convert_payload(payload)
    assert converted == {
        str(Path("foo") / "bar"): "path key",
        BaseNameGenerator.Naming.NUMBERED.value: "enum key",
        1.5: "numpy scalar key",
        # A converted tuple is a list, which cannot be a mapping key, hence
        # the string representation of the original key.
        "(1, 2)": "tuple key",
    }


def test_yaml_conversion_handles_string_like_values(convert_payload):
    """Verify that a string-like value is converted to a plain `str`.

    PyYAML represents a value by its exact type, so the representer registered
    for `str` would not be found for a subclass such as `numpy.str_`. A
    `UserString` is not even a `str`: it is a `Sequence` whose items are
    `UserString` instances, so converting it as a sequence would recurse until
    the stack is exhausted.
    """
    payload = {"numpy": str_("other"), "user_string": _CustomStr("text")}
    converted = convert_payload(payload)
    assert converted == {"numpy": "other", "user_string": "text"}


def test_yaml_conversion_handles_bytes_like_values(convert_payload):
    """Verify that a bytes-like value is converted to plain `bytes`.

    For the same reason as a string-like value: PyYAML represents the binary
    scalar for `bytes` only, so a subclass such as `numpy.bytes_` would find no
    representer and make the dump raise.
    """
    payload = {"plain": b"raw", "numpy": bytes_(b"other")}
    converted = convert_payload(payload)
    assert converted == {"plain": b"raw", "numpy": b"other"}


def test_yaml_conversion_handles_unrepresentable_arrays(convert_payload):
    """Verify the conversion of arrays whose items PyYAML cannot represent.

    A complex, object or extended-precision floating-point array is not on the
    fast path of the conversion: `numpy.ndarray.tolist` leaves a `complex`, an
    arbitrary object and a `numpy.longdouble` respectively, none of which
    PyYAML has a representer for, so returning that list as it is would make
    the dump raise and lose the whole trace. The payload is read back from the
    written file, so this test fails if the dump raises.
    """
    payload = {
        "complex": array([1.0 + 2.0j]),
        "object": array([{"key": 1}], dtype=object),
        "long_double": array([1.5], dtype=longdouble),
        "long_double_scalar": longdouble(1.5),
        "datetime": array(["2020-01-01"], dtype="datetime64[us]"),
    }
    converted = convert_payload(payload)
    # A complex array is cast to strings as a whole, giving what the per-item
    # `str` fallback would.
    assert converted["complex"] == ["(1+2j)"]
    # An object array is converted item by item, an item possibly holding a
    # container of its own.
    assert converted["object"] == [{"key": 1}]
    # `numpy.longdouble.item` returns a `numpy.longdouble` again, no Python
    # type holding its precision, hence the `str` fallback.
    assert converted["long_double"] == ["1.5"]
    assert converted["long_double_scalar"] == "1.5"
    # A datetime array is on the fast path: its items are plain `datetime`s.
    assert converted["datetime"] == [datetime(2020, 1, 1)]


def test_yaml_conversion_of_a_date_does_not_depend_on_the_array_form(convert_payload):
    """Verify that a date converts the same way inside an array and as a scalar.

    A `datetime64` array of a unit of a day or coarser is on the fast path,
    where `numpy.ndarray.tolist` gives `datetime.date` items; the very same
    value met as a `numpy.datetime64` scalar unwraps to a `datetime.date` too.
    Both forms must therefore reach the same YAML scalar, otherwise
    `yaml.safe_load` reads back a `datetime.date` in one case and a `str` in the
    other for one and the same date.
    """
    payload = {
        "array": array(["2020-01-02"], dtype="datetime64[D]"),
        "scalar": datetime64("2020-01-02"),
    }
    converted = convert_payload(payload)
    assert converted == {"array": [date(2020, 1, 2)], "scalar": date(2020, 1, 2)}


def test_yaml_conversion_keeps_a_small_array_inline_and_references_a_large_one(
    tracer, tmp_path
):
    """Verify the `_max_inline_array_size` threshold between inline and referenced.

    An array of exactly `_max_inline_array_size` elements is still converted
    inline; one more element crosses the threshold, and is written to a
    sibling `.npy` file instead, referenced by a mapping.
    """
    small_array = arange(_max_inline_array_size, dtype=float)
    large_array = arange(_large_size, dtype=float)
    call_arguments = {"end_trace": {"small": small_array, "large": large_array}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["small"] == small_array.tolist()
    assert trace["large"] == {
        "npy": ".gemseo-trace.arrays/0.npy",
        "dtype": str(large_array.dtype),
        "shape": [_large_size],
    }
    array_path = tmp_path / ".gemseo-trace.arrays" / "0.npy"
    assert array_equal(load_npy(array_path), large_array)


def test_yaml_conversion_keeps_an_object_array_inline_whatever_its_size(
    convert_payload, tmp_path
):
    """Verify that an object-dtype array is never written to a `.npy` file.

    `numpy.save` would need `allow_pickle=True` to store it, which this module
    refuses since unpickling executes arbitrary code; such an array is
    converted inline instead, whatever its size.
    """
    items = [{"k": i} for i in range(_large_size)]
    payload = {"value": array(items, dtype=object)}
    converted = convert_payload(payload)
    assert converted["value"] == items
    assert not (tmp_path / ".gemseo-trace.arrays").exists()


@pytest.mark.parametrize(
    "array_value",
    [
        arange(_large_size, dtype=float),
        arange(_large_size),
        arange(_large_size) + 1j * arange(_large_size),
        arange(_large_size).astype("datetime64[D]"),
        arange(2 * _large_size).reshape(_large_size, 2),
        array(
            [(i, float(i)) for i in range(_large_size)],
            dtype=[("a", "i4"), ("b", "f8")],
        ),
    ],
    ids=["float", "int", "complex", "datetime64", "2d", "structured"],
)
def test_load_yaml_trace_round_trips_a_large_array(tracer, tmp_path, array_value):
    """Verify that `load_yaml_trace` reads back a referenced array unchanged."""
    call_arguments = {"end_trace": {"value": array_value}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    loaded = load_yaml_trace(tmp_path / ".gemseo-trace.yml")
    assert array_equal(loaded["value"], array_value)
    assert loaded["value"].dtype == array_value.dtype


def test_start_and_end_arrays_get_distinct_files_in_the_same_directory(
    tracer, tmp_path
):
    """Verify that the start and end arrays of one call do not collide.

    `start` and `end` each convert with the same array store, created once per
    call cycle, so the index it hands out keeps increasing across the two
    calls, instead of both writing to `0.npy`.
    """
    start_array = arange(_large_size, dtype=float)
    end_array = start_array + _large_size
    call_arguments = {
        "start_trace": {"start_value": start_array},
        "end_trace": {"end_value": end_array},
    }

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["start_value"]["npy"] == ".gemseo-trace.arrays/0.npy"
    assert trace["end_value"]["npy"] == ".gemseo-trace.arrays/1.npy"
    arrays_directory = tmp_path / ".gemseo-trace.arrays"
    assert array_equal(load_npy(arrays_directory / "0.npy"), start_array)
    assert array_equal(load_npy(arrays_directory / "1.npy"), end_array)


def test_start_array_snapshot_is_immune_to_a_later_in_place_mutation(tracer, tmp_path):
    """Verify that a large start array is written before the observed call runs.

    `start` writes the arrays met while converting the start trace
    immediately, before the observed call runs: an observed call mutating its
    argument in place, e.g. a discipline adding to its input array, must not
    change the `.npy` file already written, mirroring
    `test_start_trace_is_converted_when_it_is_captured` for the inline case.
    """
    original_value = arange(_large_size, dtype=float)
    mutated_value = original_value.copy()
    call_arguments = {"start_trace": {"value": mutated_value}}

    tracer.start(call_arguments, tmp_path)
    mutated_value += 100.0
    tracer.end(call_arguments, None, tmp_path)

    array_path = tmp_path / ".gemseo-trace.arrays" / "0.npy"
    assert array_equal(load_npy(array_path), original_value)


def test_no_arrays_directory_is_created_without_a_large_array(tracer, tmp_path):
    """Verify that no arrays directory is created when no array is large."""
    call_arguments = {"end_trace": {"value": array([1.0, 2.0])}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    assert not (tmp_path / ".gemseo-trace.arrays").exists()


def test_yaml_conversion_normalizes_scalar_subclasses(convert_payload):
    """Verify that a subclass of a scalar type is normalized to a plain value.

    PyYAML resolves a representer by the exact type of a value, so it would
    find none for a subclass of `int` or of `datetime`, e.g. a
    `pandas.Timestamp`, and the dump would raise, losing the whole trace.
    """
    payload = {"integer": _CustomInt(3), "timestamp": Timestamp("2020-01-01")}
    converted = convert_payload(payload)
    # A `datetime` subclass is traced by its ISO 8601 string rather than by a
    # rebuilt plain `datetime`, since it may carry a precision a `datetime`
    # cannot hold, e.g. the nanoseconds of a `pandas.Timestamp`.
    assert converted == {"integer": 3, "timestamp": "2020-01-01T00:00:00"}
    assert type(converted["integer"]) is int


def test_end_leaves_no_trace_file_when_the_dump_raises(tracer, tmp_path, monkeypatch):
    """Verify that a failed dump leaves neither a trace file nor a temporary one.

    The trace is dumped to a sibling temporary file before being moved into
    place, so a dump that raises leaves no `.gemseo-trace.yml` at all, rather than an
    empty one that `yaml.safe_load` would read back as `None`, and no leftover
    temporary file either. Mirrors
    `test_register_leaves_no_temporary_file_when_the_dump_raises` for the
    registry writer, both going through `dump_yaml_to_file`.
    """

    def _raise_on_dump(data: Any, stream: Any) -> None:
        """Fail the dump, as a value with no representer would.

        Args:
            data: The data to dump, ignored.
            stream: The stream to dump the data to, ignored.

        Raises:
            yaml.representer.RepresenterError: Always.
        """
        msg = "cannot represent an object"
        raise yaml.representer.RepresenterError(msg)

    monkeypatch.setattr(_yaml, "dump_yaml", _raise_on_dump)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    with pytest.raises(yaml.representer.RepresenterError):
        tracer.end(call_arguments, None, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_end_leaves_no_temporary_file_when_the_move_raises(
    tracer, tmp_path, monkeypatch
):
    """Verify that a failed move of the temporary file leaves nothing behind.

    The move onto the trace path can fail on its own, e.g. with a
    `PermissionError` raised on Windows when a tool consuming the traces while
    the run is ongoing holds the target file open. The temporary file must be
    removed then as well, since a reader is promised either no file at all or a
    complete one.
    """

    def _raise_on_replace(self: Path, target: Any) -> None:
        """Fail the move, as a target file open by another process would.

        Args:
            self: The temporary file path, ignored.
            target: The path to move the file onto, ignored.

        Raises:
            PermissionError: Always.
        """
        msg = "the target file is used by another process"
        raise PermissionError(msg)

    monkeypatch.setattr(Path, "replace", _raise_on_replace)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    with pytest.raises(PermissionError):
        tracer.end(call_arguments, None, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_yaml_conversion_handles_design_spaces_and_settings(convert_payload):
    """Verify the conversion of a design space and of settings.

    Both are constructor arguments of a scenario, hence traced in its registry
    entry; the `str` fallback would give a pretty-printed table for the former
    and a repr for the latter, from which nothing can be read back.
    """
    design_space = SellarDesignSpace()
    payload = {"design_space": design_space, "settings": MDF_Settings()}
    converted = convert_payload(payload)
    assert converted["design_space"] == list(design_space.variables)
    settings = converted["settings"]
    # The name of the class configured by the settings, i.e. the formulation.
    assert settings["target_class_name"] == "MDF"
    # A field holding settings is converted by this same branch, so the class
    # it configures is known at every level.
    assert settings["main_mda_settings"]["target_class_name"] == "MDAChain"


def test_yaml_conversion_of_a_model_that_is_not_settings(convert_payload):
    """Verify that a pydantic model which is not settings has no target class name.

    Only settings configure a class, hence `target_class_name` shall not be
    added to the fields of any other model.
    """
    payload = {"model": _PlainModel(value=2.0)}
    converted = convert_payload(payload)
    assert converted["model"] == {"value": 2.0}


def test_discipline_execution_tracer_reads_input_data_from_kwarg(tmp_path):
    """Verify that the input data is read from the `input_data` keyword argument."""
    discipline = DummyDiscipline(input_names=["x", "y"], output_names=["z"])
    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    input_data = {"x": array([1.0]), "y": array([2.0])}
    call_arguments = {"input_data": input_data}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"z": array([3.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["type"] == "DummyDiscipline"
    assert trace["input_data"] == {"x": [1.0], "y": [2.0]}
    assert trace["output_data"] == {"z": [3.0]}


def test_discipline_execution_tracer_reads_input_data_from_positional_arg(tmp_path):
    """Verify that the input data passed positionally is read.

    The call specification normalizes the arguments of `Discipline.execute`, so
    the tracer reads the input data by parameter name in that case too.
    """
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    input_data = {"x": array([1.0])}
    call_arguments = normalize_arguments_safely(Discipline.execute, (input_data,), {})

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"y": array([2.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["input_data"] == {"x": [1.0]}


def test_discipline_execution_tracer_traces_no_input_data_without_call_arguments(
    tmp_path,
):
    """Verify that no input data is traced when the call passes none.

    The call specification then holds the empty default value of the
    `input_data` parameter of `Discipline.execute`; the discipline resolves its
    default input data after the observation has started.
    """
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    # `input_data`/`output_data` are real (empty) stores from construction, so
    # this sets the discipline's current data without a full `execute()`
    # (whose output grammar validation would fail: `_run` produces nothing).
    discipline.io.input_data["x"] = array([1.0])

    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    call_arguments = normalize_arguments_safely(Discipline.execute, (), {})

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {}, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["input_data"] == {}


def test_discipline_execution_tracer_traces_no_input_data_when_absent(tmp_path):
    """Verify that a call missing the `input_data` parameter loses it, without raising.

    The `input_data` parameter may simply be absent from the call arguments,
    e.g. a variadic `execute` binding it under a different name, or a
    subclass renaming it; the trace then loses the input data instead of
    breaking the observed call.
    """
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"y": array([2.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["input_data"] == {}


def test_discipline_execution_tracer_filters_input_and_output_by_grammar(tmp_path):
    """Verify that data items outside the discipline's grammars are dropped."""
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    call_arguments = {"input_data": {"x": array([1.0]), "unrelated": array([9.0])}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"y": array([2.0]), "extra": array([9.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["input_data"] == {"x": [1.0]}
    assert trace["output_data"] == {"y": [2.0]}


def test_discipline_execution_tracer_preserves_the_key_order_of_the_call_data(
    tmp_path,
):
    """Verify that the traced input data preserves the call data's key order.

    `_filter_data` iterates over the data's keys, not over a set intersection
    with the grammar's names, so the key order in `.gemseo-trace.yml` does not vary
    across processes because of string hash randomization.
    """
    discipline = DummyDiscipline(input_names=["a", "b"], output_names=["y"])
    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    input_data = {"b": array([2.0]), "a": array([1.0])}
    call_arguments = {"input_data": input_data}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"y": array([3.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert list(trace["input_data"]) == ["b", "a"]


def test_discipline_execution_tracer_records_the_input_data_of_the_start(tmp_path):
    """Verify that the traced input data is the one the observed call started with.

    A discipline mutating its input array in place would otherwise be traced
    with the value the array had once the execution was over.
    """
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    input_data = {"x": array([1.0])}
    call_arguments = {"input_data": input_data}

    tracer.start(call_arguments, tmp_path)
    input_data["x"] += 100.0
    tracer.end(call_arguments, {"y": array([2.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["input_data"] == {"x": [1.0]}


def test_discipline_execution_tracer_records_hdf5_cache_path(tmp_path):
    """Verify that the HDF5 cache file path is recorded for an `HDF5Cache`."""
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    hdf_file_path = tmp_path / "cache.hdf5"
    discipline.set_cache(discipline.CacheType.HDF5, hdf_file_path=hdf_file_path)
    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    call_arguments = {"input_data": {"x": array([1.0])}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"y": array([2.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["cache_path"] == str(hdf_file_path)


def test_discipline_execution_tracer_omits_cache_path_for_non_hdf5_cache(tmp_path):
    """Verify that no cache path is recorded for a non-`HDF5Cache` cache."""
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    discipline.set_cache(discipline.CacheType.MEMORY_FULL)
    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    call_arguments = {"input_data": {"x": array([1.0])}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"y": array([2.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert "cache_path" not in trace


def test_discipline_execution_tracer_falls_back_to_output_data_when_no_returned_data(
    tmp_path,
):
    """Verify that `_get_end_trace` tolerates a missing return value.

    `injector._end_observation_safely` always passes `returned_data=None`
    when the observed callable raised; the discipline's current output data
    is used instead, reflecting whatever was produced before the failure.
    """
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    discipline.io.output_data["y"] = array([5.0])
    tracer = DisciplineExecutionTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    call_arguments = {"input_data": {"x": array([1.0])}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["output_data"] == {"y": [5.0]}


def test_discipline_linearization_tracer_records_input_data_and_jacobian(tmp_path):
    """Verify that the trace records the input data and the returned Jacobian."""
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    tracer = DisciplineLinearizationTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    call_arguments = {"input_data": {"x": array([1.0])}}
    jacobian = {"y": {"x": array([[2.0]])}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, jacobian, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["type"] == "DummyDiscipline"
    assert trace["input_data"] == {"x": [1.0]}
    assert trace["jacobian"] == {"y": {"x": [[2.0]]}}


def test_discipline_linearization_tracer_stores_none_jacobian_without_raising(
    tmp_path,
):
    """Verify that `_get_end_trace` tolerates a `None` Jacobian, e.g. on failure."""
    discipline = DummyDiscipline(input_names=["x"], output_names=["y"])
    tracer = DisciplineLinearizationTracer(
        SimpleNamespace(object_=discipline), _no_init_arguments
    )
    call_arguments = {"input_data": {"x": array([1.0])}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["jacobian"] is None


def test_mda_execution_tracer_reuses_discipline_execution_logic(tmp_path):
    """Verify that the MDA execution tracer reuses the discipline execution logic."""
    mda = DummyDiscipline(input_names=["x"], output_names=["y"])
    tracer = MDAExecutionTracer(SimpleNamespace(object_=mda), _no_init_arguments)
    call_arguments = {"input_data": {"x": array([1.0])}}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"y": array([2.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["type"] == "DummyDiscipline"
    assert trace["input_data"] == {"x": [1.0]}
    assert trace["output_data"] == {"y": [2.0]}


def test_mda_iteration_tracer_records_iteration_number(tmp_path):
    """Verify that the iteration tracer adds the current iteration counter."""
    mda = DummyDiscipline(input_names=["x"], output_names=["y"])
    mda._current_iter = 4
    tracer = MDAIterationTracer(SimpleNamespace(object_=mda), _no_init_arguments)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"y": array([2.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["type"] == "DummyDiscipline"
    assert trace["iteration"] == 4


def test_mda_iteration_tracer_reads_input_data_from_the_solver(tmp_path):
    """Verify that the iteration input data comes from the solver's current data.

    The observed method `_iterate_once` takes no argument, and the coupling
    values of the running iteration are on the output side of the solver's
    data, hence the merged data.
    """
    mda = DummyDiscipline(input_names=["x", "y"], output_names=["y"])
    mda._current_iter = 1
    # `input_data`/`output_data` are real (empty) stores from construction, so
    # this sets the solver's current data without a full `execute()`.
    mda.io.input_data.update({"x": array([1.0]), "y": array([2.0])})
    mda.io.output_data["y"] = array([3.0])

    tracer = MDAIterationTracer(SimpleNamespace(object_=mda), _no_init_arguments)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, {"y": array([3.0])}, tmp_path)

    trace = _load_trace(tmp_path)
    # The output-side `y` overrides the input-side one, `x` is unchanged.
    assert trace["input_data"] == {"x": [1.0], "y": [3.0]}


def test_optimizer_tracer_reads_iteration_from_observer(tmp_path):
    """Verify that the current iteration is read from the observer when available."""
    tracer = OptimizerTracer(
        SimpleNamespace(object_=_DocumentedObservee(), iteration=5),
        _no_init_arguments,
    )
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["iteration"] == 5


def test_optimizer_tracer_omits_iteration_when_unavailable(tmp_path):
    """Verify that the iteration key is omitted, not an error, when unavailable.

    The observer has no iteration number until the observation of `execute` has
    captured the evaluation counter of the problem.
    """
    tracer = OptimizerTracer(
        SimpleNamespace(object_=_DocumentedObservee(), iteration=None),
        _no_init_arguments,
    )
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert "iteration" not in trace


def test_doe_tracer_records_sample_index_and_input_value_from_positional_arg(
    tmp_path,
):
    """Verify that the sample index and input value are read from the call spec."""
    tracer = DOETracer(
        SimpleNamespace(object_=_DocumentedObservee()), _no_init_arguments
    )
    input_value = array([1.0, 2.0])
    call_arguments = normalize_arguments_safely(
        BaseDOELibrary._evaluate_functions, (input_value,), {"sample_index": 3}
    )

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["type"] == "_DocumentedObservee"
    assert trace["sample_index"] == 3
    assert trace["input_value"] == [1.0, 2.0]


def test_doe_tracer_reads_input_value_from_kwarg(tmp_path):
    """Verify that the input value passed as a keyword argument is read."""
    tracer = DOETracer(
        SimpleNamespace(object_=_DocumentedObservee()), _no_init_arguments
    )
    input_value = array([4.0])
    call_arguments = normalize_arguments_safely(
        BaseDOELibrary._evaluate_functions, (), {"input_value": input_value}
    )

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["input_value"] == [4.0]


def test_doe_tracer_defaults_sample_index_when_absent(tmp_path):
    """Verify that a negative sample index means all samples evaluated at once."""
    tracer = DOETracer(
        SimpleNamespace(object_=_DocumentedObservee()), _no_init_arguments
    )
    call_arguments = normalize_arguments_safely(
        BaseDOELibrary._evaluate_functions, (array([4.0]),), {}
    )

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["sample_index"] == -1


def test_doe_tracer_defaults_when_the_call_arguments_are_absent(tmp_path):
    """Verify that call arguments missing both parameters lose the input value.

    `sample_index` and `input_value` may simply be absent from the call
    arguments, e.g. a variadic method binding them under different names, or
    a subclass renaming them; the trace then falls back to the sample index
    default and drops the input value instead of breaking the observed call.
    """
    tracer = DOETracer(
        SimpleNamespace(object_=_DocumentedObservee()), _no_init_arguments
    )
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["sample_index"] == -1
    assert "input_value" not in trace


class _FakeDatabase:
    """A fake optimization database exposing only `get_iteration`."""

    def __init__(self, iteration: int | None) -> None:
        self.__iteration = iteration

    def get_iteration(self, design: Any) -> int:
        """Return the fixed fake iteration, ignoring `design`.

        Args:
            design: The design vector, ignored.

        Returns:
            The fixed fake iteration.

        Raises:
            KeyError: When the fake iteration is `None`, as the real database
                does for a design vector it does not hold.
        """
        if self.__iteration is None:
            raise KeyError(design)
        return self.__iteration


class _FakeOptimizationProblem(OptimizationProblem):
    """An optimization problem with a fake objective, optimum and database.

    `ScenarioTracer` tells an optimization problem from an evaluation one by
    its type, hence the real base class; everything the tracer reads from it is
    faked, so that no function has to be evaluated.
    """

    def __init__(
        self,
        objective_name: str = "obj",
        optimum: Any = None,
        iteration: int | None = 0,
    ) -> None:
        """
        Args:
            objective_name: The name of the objective.
                If empty, no objective is set, as before
                `MDOScenario.add_objective` has been called.
            optimum: The optimum. If `None`, reading it raises `ValueError`.
            iteration: The iteration at which the optimum was found.
                If `None`, looking it up raises `KeyError`.
        """  # noqa: D205, D212
        super().__init__(DesignSpace())
        if objective_name:
            # `OptimizationProblem.objective_name` reads the name of the
            # objective function, which is all the tracer needs.
            self._objective = SimpleNamespace(name=objective_name)
        self.database = _FakeDatabase(iteration)
        self.__optimum = optimum

    @property
    def optimum(self) -> Any:
        """Return the optimum, or raise like `OptimizationProblem.optimum` does."""
        if self.__optimum is None:
            msg = "The optimization history is empty."
            raise ValueError(msg)
        return self.__optimum


def _make_scenario_observee(
    problem: Any, variable_names: tuple[str, ...] = ("x",)
) -> Any:
    """Build a fake scenario exposing what `ScenarioTracer` reads.

    Args:
        problem: The formulation's problem.
        variable_names: The names of the design variables.

    Returns:
        A fake scenario.
    """
    return SimpleNamespace(
        formulation_name="MDF",
        design_space=SimpleNamespace(variable_names=list(variable_names)),
        formulation=SimpleNamespace(problem=problem),
    )


def test_scenario_tracer_records_objective(tmp_path):
    """Verify that the end trace records the objective.

    The formulation and the design variables are not traced per execution:
    they follow from the `formulation_settings` and `design_space` constructor
    arguments, traced in the registry entry (see
    `test_yaml_conversion_handles_design_spaces_and_settings`).
    """
    problem = _FakeOptimizationProblem(objective_name="obj")
    scenario = _make_scenario_observee(problem)
    tracer = ScenarioTracer(SimpleNamespace(object_=scenario), _no_init_arguments)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["objective"] == "obj"
    assert "formulation" not in trace
    assert "design_variables" not in trace


def test_scenario_tracer_omits_objective_for_plain_evaluation_problem(tmp_path):
    """Verify that a plain evaluation problem, with no objective, omits the key."""
    scenario = _make_scenario_observee(EvaluationProblem(DesignSpace()))
    tracer = ScenarioTracer(SimpleNamespace(object_=scenario), _no_init_arguments)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert "objective" not in trace


def test_scenario_tracer_omits_objective_when_it_is_not_set_yet(tmp_path):
    """Verify that an optimization problem with no objective omits both keys.

    An `MDOScenario` has an `OptimizationProblem` from the start, but its
    objective is only set by a later call to `MDOScenario.add_objective`; both
    `objective_name` and `optimum` read the objective function, so neither can
    be traced before then.
    """
    problem = _FakeOptimizationProblem(objective_name="")
    scenario = _make_scenario_observee(problem)
    tracer = ScenarioTracer(SimpleNamespace(object_=scenario), _no_init_arguments)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert "objective" not in trace
    assert "optimum" not in trace


def test_scenario_tracer_records_optimum_objective_and_iteration(tmp_path):
    """Verify that the end trace records the optimum's objective and iteration."""
    optimum = SimpleNamespace(objective=3.5, design="x_opt")
    problem = _FakeOptimizationProblem(optimum=optimum, iteration=7)
    scenario = _make_scenario_observee(problem)
    tracer = ScenarioTracer(SimpleNamespace(object_=scenario), _no_init_arguments)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert trace["optimum"] == {"objective": 3.5, "iteration": 7}


def test_scenario_tracer_omits_optimum_when_database_is_empty(tmp_path):
    """Verify that an empty database (raising `ValueError`) omits `optimum`."""
    problem = _FakeOptimizationProblem(optimum=None)
    scenario = _make_scenario_observee(problem)
    tracer = ScenarioTracer(SimpleNamespace(object_=scenario), _no_init_arguments)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert "optimum" not in trace


def test_scenario_tracer_omits_optimum_when_its_design_is_not_in_the_database(tmp_path):
    """Verify that a design vector absent from the database omits `optimum`.

    `OptimizationHistory.optimum` returns an empty design vector when no
    feasible point carries the value of the objective, and
    `Database.get_iteration` raises `KeyError` for such a vector.
    """
    optimum = SimpleNamespace(objective=3.5, design=array([]))
    problem = _FakeOptimizationProblem(optimum=optimum, iteration=None)
    scenario = _make_scenario_observee(problem)
    tracer = ScenarioTracer(SimpleNamespace(object_=scenario), _no_init_arguments)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert "optimum" not in trace
    # The objective name does not depend on the optimum.
    assert trace["objective"] == "obj"


def test_scenario_tracer_omits_optimum_for_plain_evaluation_problem(tmp_path):
    """Verify that a plain evaluation problem, with no `optimum`, omits the key.

    This is a real, non-optimization case: `gemseo.sample_disciplines` builds
    a plain `EvaluationScenario` whose problem is an `EvaluationProblem`,
    which has no `optimum` attribute at all (unlike `OptimizationProblem`,
    which defines it). This is what the type check of `_get_end_trace` is for.
    """
    scenario = _make_scenario_observee(EvaluationProblem(DesignSpace()))
    tracer = ScenarioTracer(SimpleNamespace(object_=scenario), _no_init_arguments)
    call_arguments = {}

    tracer.start(call_arguments, tmp_path)
    tracer.end(call_arguments, None, tmp_path)

    trace = _load_trace(tmp_path)
    assert "optimum" not in trace


@pytest.fixture
def dm_settings(tmp_wd: Path):
    """Enable and reset the directory manager for the duration of a test.

    The manager cannot be disabled once enabled, so the previous (disabled)
    settings instance is restored on teardown instead of toggling `enable`.
    """
    previous_settings = _configuration.directory_manager
    settings = _configuration.directory_manager = Settings()
    settings.enable = True
    settings.execution_root_path = tmp_wd / "root"
    yield settings
    _configuration.directory_manager = previous_settings


def test_mdf_scenario_execution_writes_every_tracer_type(
    dm_settings, sellar_disciplines
):
    """Verify that a real MDF+SLSQP Sellar run writes a parseable trace per type."""
    design_space = SellarDesignSpace()
    scenario = create_scenario(
        list(sellar_disciplines),
        "obj",
        design_space,
        formulation_settings_model=MDF_Settings(),
    )
    scenario.add_constraint("c_1", constraint_type=scenario.ConstraintType.INEQ)
    scenario.add_constraint("c_2", constraint_type=scenario.ConstraintType.INEQ)
    scenario.execute(SLSQP_Settings(max_iter=2))

    root_path = dm_settings.execution_root_path
    trace_paths = list(root_path.rglob(".gemseo-trace.yml"))
    assert trace_paths

    registry_root_path = root_path / ".gemseo-traces"
    traces = [yaml.safe_load(trace_path.read_text()) for trace_path in trace_paths]
    for trace in traces:
        assert isinstance(trace, dict)
        assert "object_id" in trace
        assert "type" in trace
        # Every per-call object_id resolves to an existing registry entry.
        registry_path = registry_root_path / f"{trace['object_id']}.trace.yml"
        assert registry_path.is_file()

    # The scenario itself is stamped with the object_id of its own registry
    # entry (see `BaseTracer.__resolve_object_id`), so its trace is found the
    # same way a caller resolving the id from a trace file would.
    scenario_trace = next(
        trace for trace in traces if trace["object_id"] == scenario._workflow_trace_id
    )
    assert "optimum" in scenario_trace
    assert "objective" in scenario_trace["optimum"]
    assert "iteration" in scenario_trace["optimum"]

    # A discipline's execution and linearization traces share one object_id:
    # `DisciplineWorkflowObserver` builds both tracers for the same instance
    # (see `BaseWorkflowObserverDispatcher.__init__`), so the id stamped by
    # whichever runs first is reused by the other. An execution trace is
    # identified by its `output_data` key, a linearization one by `jacobian`.
    shares_one_object_id = False
    for discipline in sellar_disciplines:
        discipline_traces = [
            trace
            for trace in traces
            if trace["object_id"] == discipline._workflow_trace_id
        ]
        has_execution = any("output_data" in trace for trace in discipline_traces)
        has_linearization = any("jacobian" in trace for trace in discipline_traces)
        if has_execution and has_linearization:
            shares_one_object_id = True
            break
    assert shares_one_object_id

    # A registry entry carries the documentation and init_arguments moved
    # out of the per-call traces.
    registry_paths = list(registry_root_path.rglob("*.trace.yml"))
    assert registry_paths
    for registry_path in registry_paths:
        entry = yaml.safe_load(registry_path.read_text())
        assert "documentation" in entry
        assert "init_arguments" in entry

    # The formulation and the design variables of the scenario are read from
    # its constructor arguments, instead of being traced per execution.
    scenario_entry = yaml.safe_load(
        (registry_root_path / f"{scenario._workflow_trace_id}.trace.yml").read_text()
    )
    init_arguments = scenario_entry["init_arguments"]
    assert init_arguments["design_space"] == list(design_space.variables)
    assert init_arguments["formulation_settings"]["target_class_name"] == "MDF"


def test_doe_scenario_execution_writes_one_doe_trace_per_sample(
    dm_settings, sellar_disciplines
):
    """Verify that a real DOE Sellar run writes one DOE trace per sample."""
    design_space = SellarDesignSpace()
    scenario = create_scenario(
        list(sellar_disciplines),
        "obj",
        design_space,
        scenario_type="DOE",
        formulation_settings_model=MDF_Settings(),
    )
    scenario.execute(LHS_Settings(n_samples=3))

    trace_paths = list(dm_settings.execution_root_path.rglob(".gemseo-trace.yml"))
    traces = [yaml.safe_load(trace_path.read_text()) for trace_path in trace_paths]
    # A DOE trace is identified by its `sample_index` key: the DOE library's
    # real class name (`type`) is not hard-coded here.
    doe_traces = [trace for trace in traces if "sample_index" in trace]

    assert len(doe_traces) == 3
    sample_indices = set()
    for trace in doe_traces:
        assert "sample_index" in trace
        assert trace["input_value"]
        sample_indices.add(trace["sample_index"])
    assert sample_indices == {0, 1, 2}
