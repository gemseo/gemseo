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

"""Tests for the trace registry."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import yaml
from numpy import arange
from numpy import array_equal
from numpy import load as load_npy

from gemseo.core.grammar.factory import GrammarFactory
from gemseo.util import _worker_context
from gemseo.util._directory_manager.settings import Settings
from gemseo.util._tracer import _yaml
from gemseo.util._tracer._yaml import _max_inline_array_size
from gemseo.util._tracer.registry import TraceRegistry
from gemseo.util.base_multiton import BaseMultiton
from gemseo.util.global_configuration import _configuration


@pytest.fixture(autouse=True)
def _reset_trace_registry():
    """Ensure that every test starts from, and leaves, an empty registry.

    Without this, the per-class counters would carry over from whatever
    other test happened to run first in the same process, making the exact
    indices asserted below non-deterministic.
    """
    BaseMultiton.clear_cache(TraceRegistry)
    yield
    BaseMultiton.clear_cache(TraceRegistry)


@pytest.fixture
def dm_settings(tmp_wd):
    """Enable and reset the directory manager for the duration of a test.

    Only used by the test verifying that the `enable` validator resets the
    trace registry's cache: this is the one place where the directory
    manager settings still reach directly into `TraceRegistry`, independently
    of the `set_root_path` call performed by `BaseDMProcessor`.

    The manager cannot be disabled once enabled, so the previous (disabled)
    settings instance is restored on teardown instead of toggling `enable`.
    """
    previous_settings = _configuration.directory_manager
    settings = _configuration.directory_manager = Settings()
    settings.enable = True
    settings.execution_root_path = tmp_wd / "root"
    yield settings
    _configuration.directory_manager = previous_settings


def test_register_returns_sequential_ids_and_writes_entries_when_root_path_is_set(
    tmp_path,
):
    """Verify sequential per-class indices, and the written entries' content."""
    registry = TraceRegistry()
    registry.set_root_path(tmp_path)

    first_id = registry.register(
        "Foo",
        name="alpha",
        documentation="Foo's documentation.",
        init_arguments={"args": [], "kwargs": {}},
    )
    second_id = registry.register(
        "Foo",
        name="",
        documentation="Foo's documentation.",
        init_arguments={"args": [1], "kwargs": {}},
    )

    assert first_id == "Foo/0"
    assert second_id == "Foo/1"

    registry_directory = tmp_path / ".gemseo-traces" / "Foo"
    first_entry = yaml.safe_load((registry_directory / "0.trace.yml").read_text())
    assert first_entry == {
        "object_id": "Foo/0",
        "type": "Foo",
        "name": "alpha",
        "documentation": "Foo's documentation.",
        "init_arguments": {"args": [], "kwargs": {}},
    }
    # The key order matches the decided registry schema.
    assert list(first_entry.keys()) == [
        "object_id",
        "type",
        "name",
        "documentation",
        "init_arguments",
    ]

    second_entry = yaml.safe_load((registry_directory / "1.trace.yml").read_text())
    assert second_entry["object_id"] == "Foo/1"
    assert second_entry["init_arguments"] == {"args": [1], "kwargs": {}}
    # `name` is only written when the observee has one.
    assert "name" not in second_entry
    assert list(second_entry.keys()) == [
        "object_id",
        "type",
        "documentation",
        "init_arguments",
    ]


def test_register_uses_independent_counters_per_class():
    """Verify that different classes are counted independently."""
    registry = TraceRegistry()

    foo_id = registry.register("Foo", name="", documentation="", init_arguments={})
    bar_id = registry.register("Bar", name="", documentation="", init_arguments={})
    second_foo_id = registry.register(
        "Foo", name="", documentation="", init_arguments={}
    )

    assert foo_id == "Foo/0"
    assert bar_id == "Bar/0"
    assert second_foo_id == "Foo/1"


def test_register_skips_writing_until_a_root_path_is_set(tmp_wd):
    """Verify that no file is written until a root path is set.

    Only the directory manager, aware of an execution root, calls
    `set_root_path`; a tracer built directly, e.g. by another unit test, leaves
    the registry without a root and still gets a well-formed id.
    """
    registry = TraceRegistry()

    object_id = registry.register(
        "Foo", name="", documentation="doc", init_arguments={}
    )

    assert object_id == "Foo/0"
    assert not (tmp_wd / ".gemseo-traces").exists()


def test_counter_resets_when_the_directory_manager_is_re_enabled(dm_settings):
    """Verify that re-enabling the directory manager also resets the registry.

    Mirrors `test_enabling_resets_only_the_directory_manager` in
    `test_directory_manager.py`: only the trace registry's cache entry is
    evicted, not the other multitons'. This is the one behavior of
    `TraceRegistry` that the directory manager settings still reach into
    directly (see the `enable` validator in
    `gemseo.util._directory_manager.settings`), independently of the
    `set_root_path` call performed by `BaseDMProcessor`.
    """
    registry = TraceRegistry()
    registry.register("Foo", name="", documentation="", init_arguments={})
    grammar_factory = GrammarFactory()

    dm_settings.enable = True  # Re-assigning re-triggers the reset validator.

    assert TraceRegistry() is not registry
    assert GrammarFactory() is grammar_factory
    new_id = TraceRegistry().register(
        "Foo", name="", documentation="", init_arguments={}
    )
    assert new_id == "Foo/0"


@pytest.mark.parametrize(
    "in_worker_marker",
    ["parent_process", "_inheriting", "process_id"],
)
def test_register_prefixes_the_index_with_the_process_id_in_a_worker(
    tmp_path, monkeypatch, in_worker_marker
):
    """Verify that a worker process registers under a process-specific name.

    The per-class counters cannot disambiguate the objects constructed in
    worker processes sharing one `.gemseo-traces` directory, whatever the start
    method: a forked worker inherits a snapshot of the counters, a spawned
    one starts them over. Each of the three ways of recognizing a worker is
    simulated here, since none of them can be produced in-process.
    """
    registry = TraceRegistry()
    registry.set_root_path(tmp_path)

    if in_worker_marker == "parent_process":
        monkeypatch.setattr(_worker_context, "parent_process", lambda: "parent")
    elif in_worker_marker == "_inheriting":
        monkeypatch.setattr(
            _worker_context,
            "current_process",
            lambda: SimpleNamespace(_inheriting=True),
        )
    else:
        # A process id differing from the one stored at construction, as seen
        # by a forked worker inheriting this registry.
        monkeypatch.setattr(
            registry, "_TraceRegistry__process_id", os.getpid() + 1, raising=True
        )

    object_id = registry.register("Foo", name="", documentation="", init_arguments={})

    process_id = os.getpid()
    assert object_id == f"Foo/{process_id}-0"
    entry_path = tmp_path / ".gemseo-traces" / "Foo" / f"{process_id}-0.trace.yml"
    assert yaml.safe_load(entry_path.read_text())["object_id"] == object_id


def test_register_leaves_no_temporary_file_when_the_dump_raises(
    tmp_path, monkeypatch, caplog
):
    """Verify that a failed dump does not leave a temporary entry behind.

    The entry is dumped to a sibling temporary file before being moved into
    place, so that no reader can observe a half-written document. The dump
    itself is forced to fail, rather than passed an unrepresentable
    `init_arguments` value: `register` now converts them with
    `convert_to_yaml_data` before writing, which always falls back to `str`
    for a value with no other conversion rule, so no value reaches the dumper
    unrepresented any more. Mirrors
    `test_end_leaves_no_trace_file_when_the_dump_raises` for the per-call
    trace writer, both going through `dump_yaml_to_file`.
    """
    registry = TraceRegistry()
    registry.set_root_path(tmp_path)

    def _raise_on_dump(data, stream):
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

    object_id = registry.register("Foo", name="", documentation="", init_arguments={})

    assert object_id == "Foo/0"
    assert "The trace registry entry of Foo/0 could not be written." in caplog.text
    assert list((tmp_path / ".gemseo-traces" / "Foo").iterdir()) == []


def test_register_propagates_a_conversion_error_before_writing_anything(tmp_path):
    """Verify that a conversion error aborts registration instead of degrading it.

    The conversion of `init_arguments` happens outside the block guarding the
    write: a self-referential container is not such an error (`convert_to_yaml_data`
    cuts the cycle with a marker), but a raising `__str__` is, and it must
    propagate instead of merely being logged, so that the caller
    (`BaseTracer.__resolve_object_id`) falls back to a no-op tracer, as
    `BaseDMProcessor.__init__` does.
    """

    class _RaisingStr:
        """An object whose `__str__` raises."""

        def __str__(self) -> str:
            msg = "boom"
            raise RuntimeError(msg)

    registry = TraceRegistry()
    registry.set_root_path(tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        registry.register(
            "Foo", name="", documentation="", init_arguments={"x": _RaisingStr()}
        )

    assert not (tmp_path / ".gemseo-traces").exists()


def test_register_writes_a_large_constructor_argument_to_a_npy_file(tmp_path):
    """Verify that a large constructor argument is written next to its entry."""
    registry = TraceRegistry()
    registry.set_root_path(tmp_path)
    large_array = arange(_max_inline_array_size + 1)

    registry.register(
        "Foo", name="", documentation="", init_arguments={"x": large_array}
    )

    entry_path = tmp_path / ".gemseo-traces" / "Foo" / "0.trace.yml"
    entry = yaml.safe_load(entry_path.read_text())
    array_path = tmp_path / ".gemseo-traces" / "Foo" / "0.trace.arrays" / "0.npy"
    assert entry["init_arguments"]["x"] == {
        "npy": "0.trace.arrays/0.npy",
        "dtype": str(large_array.dtype),
        "shape": list(large_array.shape),
    }
    assert array_equal(load_npy(array_path), large_array)


def test_register_raises_on_a_conversion_error_even_without_a_root_path():
    """Verify that a conversion error still surfaces without a root path.

    Only the write is skipped when no root path was injected: the conversion
    of `init_arguments` still happens, so a conversion error still surfaces,
    matching the behavior with a root path set.
    """
    registry = TraceRegistry()

    class _RaisingStr:
        """An object whose `__str__` raises."""

        def __str__(self) -> str:
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        registry.register(
            "Foo", name="", documentation="", init_arguments={"x": _RaisingStr()}
        )


def test_register_writes_no_array_file_when_no_root_path_is_set(tmp_wd):
    """Verify that a large constructor argument writes nothing without a root path.

    Only the registry entry's write is skipped when no root path was
    injected, see `test_register_skips_writing_until_a_root_path_is_set`; this
    also holds for the `.npy` file of a large constructor argument.
    """
    registry = TraceRegistry()

    registry.register(
        "Foo",
        name="",
        documentation="",
        init_arguments={"x": arange(_max_inline_array_size + 1)},
    )

    assert list(tmp_wd.iterdir()) == []


def test_register_falls_back_to_the_raw_class_name_when_sanitizing_yields_empty():
    """Verify the fallback for a class name made only of non-ASCII characters."""
    registry = TraceRegistry()

    object_id = registry.register("中文", name="", documentation="", init_arguments={})

    assert object_id == "中文/0"


def test_register_logs_the_error_when_the_entry_cannot_be_written(tmp_path, caplog):
    """Verify that a failed write degrades the trace instead of raising.

    An observee is registered while it is being constructed, so raising here
    would abort that construction. The id is returned nonetheless, so that the
    per-call traces of the observee stay consistent with each other.
    """
    registry = TraceRegistry()
    # A file where the registry expects to create a directory: the mkdir of the
    # write raises a FileExistsError.
    (tmp_path / ".gemseo-traces").mkdir()
    (tmp_path / ".gemseo-traces" / "Foo").touch()
    registry.set_root_path(tmp_path)

    object_id = registry.register("Foo", name="", documentation="", init_arguments={})

    assert object_id == "Foo/0"
    assert "The trace registry entry of Foo/0 could not be written." in caplog.text
