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
"""The YAML conversion and dumping shared by the tracer package's writers.

Both the per-call trace writer
([BaseTracer][gemseo.util._tracer.base.BaseTracer]) and the registry writer
([TraceRegistry][gemseo.util._tracer.registry.TraceRegistry]) dump their data
through
[dump_yaml_to_file][gemseo.util._tracer._yaml.dump_yaml_to_file], which
writes a complete file or none at all and goes through
[dump_yaml][gemseo.util._tracer._yaml.dump_yaml], so that a
multi-line string, e.g. a class documentation, is emitted using the literal
block style (``|``) rather than PyYAML's default folded or double-quoted
styles, which would otherwise mangle it with escaped newline sequences and
line-continuation backslashes.

The dumper derives from PyYAML's C implementation when the installed PyYAML is
built with libyaml, which is about six times faster per dump. That
implementation stops its printability analysis at U+FFFD, so a string holding
an astral-plane character, e.g. an emoji (U+10000 and above), or one of U+0085,
U+2028 and U+2029, is emitted as an escaped double-quoted scalar instead of a
literal block. No GEMSEO source triggers this today, but user-supplied text
can.

The data is prepared by
[convert_to_yaml_data][gemseo.util._tracer._yaml.convert_to_yaml_data], which
maps the objects met in traced data, e.g. a NumPy array or a settings model,
onto the types PyYAML can represent.

A NumPy array of more than `_max_inline_array_size` elements, of a dtype kind
other than object, is not converted inline: it is handed to an
[_NpyArrayStore][gemseo.util._tracer._yaml._NpyArrayStore], which writes it to
a sibling ``.npy`` file and replaces it in the YAML data with a small mapping
referencing that file. [load_yaml_trace][gemseo.util._tracer._yaml.load_yaml_trace]
reads a trace file back and resolves such references into NumPy arrays.
"""

from __future__ import annotations

from collections import UserString
from collections.abc import Mapping
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from datetime import date
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

import numpy
import yaml
from numpy import generic
from numpy import ndarray
from pydantic import BaseModel

from gemseo.space.design import DesignSpace
from gemseo.util.pydantic import BaseSettings

if TYPE_CHECKING:
    from typing import IO


_base_dumper_class: Final[type[Any]] = (
    yaml.CSafeDumper if yaml.__with_libyaml__ else yaml.SafeDumper
)
"""The safe dumper class the tracer dumper derives from.

The C implementation is used whenever the installed PyYAML is built with
libyaml, since it dumps a trace about six times faster. The guard is required:
`yaml.CSafeDumper` does not exist at all when PyYAML is built without libyaml,
so an unguarded reference would raise an `AttributeError` when this module is
imported, breaking the whole tracer.
"""


class _TracerDumper(_base_dumper_class):
    """A safe dumper emitting multi-line strings using the literal style.

    The representer handling this is registered on this subclass only:
    `_base_dumper_class` and the rest of PyYAML's global state are never
    mutated, so no other user of PyYAML in the process is affected.
    """


def _represent_str(dumper: _TracerDumper, value: str) -> yaml.ScalarNode:
    """Represent a string, using the literal block style for multi-line ones.

    Args:
        dumper: The dumper requesting the representation.
        value: The string to represent.

    Returns:
        The scalar node representing the string.
    """
    if "\n" not in value:
        return dumper.represent_scalar("tag:yaml.org,2002:str", value)
    # PyYAML silently discards the requested style and falls back to a
    # quoted scalar when a line has trailing whitespace or the string
    # contains unprintable characters; normalizing first avoids the former
    # for the common case of a docstring (e.g. a blank line, made of
    # indentation only, before a closing indented `"""`). The latter is not
    # fought: if PyYAML still cannot use a literal block afterwards, letting
    # it fall back on its own is fine.
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", _normalize_multiline(value), style="|"
    )


def _normalize_multiline(value: str) -> str:
    """Normalize a multi-line string so PyYAML can emit it as a literal block.

    Trailing whitespace is stripped from every line, since PyYAML refuses
    the literal block style otherwise. The trailing blank line this may
    leave behind (e.g. from the indentation-only line before a closing
    indented docstring delimiter, or simply from the docstring's own
    trailing newline) is then dropped: the literal block style picks
    whichever chomping indicator reproduces the original ending on its own
    (its default "clip" chomping restores the single trailing line break of
    the common case).

    This normalization applies to every multi-line string in the dumped
    data, not only to a class documentation: a traced data value with
    trailing whitespace on one of its lines is stripped the same way, so a
    trace is not byte-faithful to the traced data. This is a deliberate
    trade-off of that byte-fidelity for the readability of the literal
    block style.

    Args:
        value: The string to normalize.

    Returns:
        The normalized string.
    """
    lines = [line.rstrip() for line in value.split("\n")]
    if lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


_TracerDumper.add_representer(str, _represent_str)

_width: Final[int] = 1_000_000
"""A width large enough that a long single-line value is never wrapped.

PyYAML would otherwise fold such a value into a multi-line scalar.
"""

_dump_kwargs: Final[Mapping[str, Any]] = MappingProxyType({
    "Dumper": _TracerDumper,
    "default_flow_style": False,
    "sort_keys": False,
    "indent": 2,
    "allow_unicode": True,
    "width": _width,
})
"""The keyword arguments passed to `yaml.dump` by `dump_yaml`."""


def dump_yaml(data: Any, stream: IO[str]) -> None:
    """Dump data as YAML, using a literal block style for multi-line strings.

    Args:
        data: The data to dump.
        stream: The writable stream to dump the data to.
    """
    yaml.dump(data, stream, **_dump_kwargs)


def dump_yaml_to_file(data: Any, file_path: Path) -> None:
    """Dump data as YAML to a file, atomically.

    The data is dumped to a sibling temporary file, which is then moved onto
    the target path, so that a reader, e.g. a tool consuming the traces while
    a run is ongoing, sees either no file at all or a complete one, never a
    half-written document. A dump that raises, e.g. on an unrepresentable
    value, therefore leaves no file behind either, rather than an empty one
    that `yaml.safe_load` would read back as `None`.

    The move costs about 25 microseconds per file, 2 to 3% of a traced run of the
    Sobieski problem, which is why it is kept rather than traded for a direct
    write; see the tracer reference documentation for the measurements.

    Args:
        data: The data to dump.
        file_path: The path of the file to dump the data to.
    """
    temporary_file_path = file_path.with_name(f"{file_path.name}.tmp")
    try:
        with temporary_file_path.open("w", encoding="utf-8") as stream:
            dump_yaml(data, stream)
        temporary_file_path.replace(file_path)
    except BaseException:
        # The leftover of a failed dump, or of a failed move, e.g. a
        # `PermissionError` raised on Windows when the target file is open by
        # another process, is removed here rather than in a `finally` clause: a
        # successful move has consumed the temporary file, which would then pay
        # a failing `unlink` syscall per written trace.
        temporary_file_path.unlink(missing_ok=True)
        raise


_representable_dtype_kinds: Final[str] = "biuMSU"
"""The NumPy dtype kinds whose items PyYAML can represent as they are.

Namely boolean, signed and unsigned integer, datetime, bytes and string:
`numpy.ndarray.tolist` unwraps such an array into `bool`, `int`,
`datetime.datetime` (or `datetime.date` for a unit of a day or coarser, or
`int` for one finer than a microsecond), `bytes` and `str` objects, all of
which PyYAML represents, so converting the items one by one would return them
unchanged, at the cost of one call per item.

The floating-point kind is admitted separately, since it is only partly
unwrapped, see `_max_unwrapped_float_itemsize`.
"""

_max_unwrapped_float_itemsize: Final[int] = 8
"""The largest floating-point dtype item size `numpy.ndarray.tolist` unwraps.

A `float16`, `float32` or `float64` array is unwrapped into `float` objects,
but a `longdouble` one (16 bytes on Linux/x86-64) keeps `numpy.longdouble`
items, which PyYAML cannot represent; its items therefore go through the
per-item conversion, which falls back to `str`.
"""

_stringifiable_dtype_kind: Final[str] = "c"
"""The NumPy dtype kind converted by casting the whole array to strings.

`numpy.ndarray.astype(str)` produces, for a complex array, exactly the strings
the per-item `str` fallback would, about four times faster. The other
unrepresentable kinds are not cast this way: the cast raises a `TypeError` for
a structured dtype (kind ``"V"``), and it spells a timedelta (kind ``"m"``) as
its raw count of units, e.g. ``"1 seconds"`` instead of ``"0:00:01"``.
"""

_plain_scalar_types: Final[frozenset[type]] = frozenset({
    bool,
    int,
    float,
    date,
    datetime,
})
"""The scalar types PyYAML represents, matched by exact type.

A subclass is not matched: PyYAML resolves a representer by the exact type of
a value, so it would find none for e.g. a `pandas.Timestamp` or a user's `int`
subclass, and the dump would raise, losing the whole trace. Such a value is
normalized instead, as a `str` subclass is above.

`date` is listed although `datetime` derives from it, since the match is on the
exact type: a `datetime` is still handled by its own branch below, while a
`date` is the object a `numpy.datetime64` of a unit of a day or coarser unwraps
to, whether it is met as a scalar or as an item of an array taking the fast path
of `numpy.ndarray.tolist`. Both forms are therefore dumped as the same YAML
date, instead of one of them reaching the `str` fallback.
"""

_cycle_marker: Final[str] = "<cycle>"
"""The marker standing for a value that contains itself.

Such a value has no YAML representation, and converting it would recurse until
the stack is exhausted.
"""

_max_inline_array_size: Final[int] = 16
"""The largest number of elements of a NumPy array converted inline.

A larger array is handed to an `_NpyArrayStore` instead, which writes it to a
`.npy` file and replaces it in the YAML data with a small reference mapping.
A small array is kept inline because it is meant to be read directly in the
YAML, alongside the rest of the trace, and because writing it to a file costs
more than converting it: a `.npy` file, with the arrays directory created for
it, costs about as much as converting an array of 8 elements inline, and a run
tracing many small vectors, e.g. the Sobieski problem whose variables have 4 to
10 elements, is about 18% slower when they are written to files. The
threshold keeps a margin above that break-even.
"""

_object_dtype_kind: Final[str] = "O"
"""The NumPy dtype kind of an array whose items are arbitrary Python objects.

Such an array is never written to a `.npy` file, whatever its size: doing so
would need `numpy.save`'s `allow_pickle=True`, which this module refuses since
unpickling executes arbitrary code; the array is converted inline instead.
"""

_array_reference_keys: Final[frozenset[str]] = frozenset({"npy", "dtype", "shape"})
"""The mapping keys identifying an array reference written by `_NpyArrayStore`.

`load_yaml_trace` replaces every mapping matching exactly this set of keys by
the array its `npy` entry points to. A user mapping that happens to have
exactly these three keys, and no other, would be read back as an array
reference too; this convention trades that unlikely collision for not needing
a wrapper tag in the YAML.
"""


class _NpyArrayStore:
    """A store for the large NumPy arrays met while converting one trace.

    Created once per conversion cycle (e.g. one per pair of
    [BaseTracer.start][gemseo.util._tracer.base.BaseTracer.start] and
    [BaseTracer.end][gemseo.util._tracer.base.BaseTracer.end] calls, or one per
    [TraceRegistry.register][gemseo.util._tracer.registry.TraceRegistry.register]
    call), and handed to `convert_to_yaml_data`. Every array larger than
    `_max_inline_array_size` met during the conversion is added to the store,
    which assigns it the next index, keeps a reference to it, and returns the
    mapping that replaces it in the YAML data. `write_arrays` then dumps every
    array kept so far to a `.npy` file and forgets it, so that a later
    conversion with the same store, e.g. `end` following `start`, starts
    indexing from where the previous conversion left off, instead of reusing a
    file name it already wrote.
    """

    __arrays_directory_name: str
    """The name of the sibling directory holding the referenced arrays."""

    __index: int
    """The index to assign to the next array added to this store."""

    __index_to_array: dict[int, ndarray]
    """The arrays added since the last call to `write_arrays`, by index."""

    def __init__(self, arrays_directory_name: str) -> None:
        """
        Args:
            arrays_directory_name: The name of the sibling directory holding
                the arrays referenced from the YAML file.
        """  # noqa: D205, D212
        self.__arrays_directory_name = arrays_directory_name
        self.__index = 0
        self.__index_to_array = {}

    def add_array(self, value: ndarray) -> dict[str, Any]:
        """Keep a large array for later writing and return its reference.

        Args:
            value: The array to keep.

        Returns:
            The mapping referencing the array's future `.npy` file.
        """
        index = self.__index
        self.__index += 1
        self.__index_to_array[index] = value
        return {
            "npy": f"{self.__arrays_directory_name}/{index}.npy",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }

    def write_arrays(self, base_directory_path: Path) -> None:
        """Write the arrays kept so far to `.npy` files, then forget them.

        The arrays directory is created only when there is at least one array
        to write.

        Args:
            base_directory_path: The directory the arrays directory name is
                relative to.
        """
        if not self.__index_to_array:
            return
        arrays_directory_path = base_directory_path / self.__arrays_directory_name
        arrays_directory_path.mkdir(parents=True, exist_ok=True)
        for index, value in self.__index_to_array.items():
            numpy.save(
                arrays_directory_path / f"{index}.npy", value, allow_pickle=False
            )
        self.__index_to_array.clear()


def convert_to_yaml_data(value: Any, array_store: _NpyArrayStore | None = None) -> Any:
    """Convert a value into a representation writable by `dump_yaml`.

    Args:
        value: The value to convert.
        array_store: The store handling the large NumPy arrays met during the
            conversion, i.e. those with more than `_max_inline_array_size`
            elements and a dtype kind other than object. When `None`, the
            default, every array is converted inline, whatever its size.

    Returns:
        The converted value.
    """
    return _convert_to_yaml_data(value, set(), array_store)


def _convert_to_yaml_data(
    value: Any, ancestor_ids: set[int], array_store: _NpyArrayStore | None
) -> Any:
    """Convert a value into a representation writable by `dump_yaml`.

    Args:
        value: The value to convert.
        ancestor_ids: The identifiers of the containers being converted on the
            path from the root value down to this one. A container records its
            own identifier there before converting its items and discards it
            afterwards, so that only a container reached from itself is cut,
            while a value met twice as a sibling, as in any plain acyclic
            structure, is converted twice as it should be.
        array_store: The store handling the large NumPy arrays met during the
            conversion, or `None` when every array must be converted inline.

    Returns:
        The converted value, or `_cycle_marker` when the value is one of the
        containers it is nested in.
    """
    if id(value) in ancestor_ids:
        return _cycle_marker
    if isinstance(value, Enum):
        # Handled before `str`, since the gemseo enums derive from `StrEnum`
        # and PyYAML represents a value by its exact type, so it would not
        # find a representer for the enum member itself.
        return _convert_to_yaml_data(value.value, ancestor_ids, array_store)
    if isinstance(value, str | UserString):
        # A `str` is converted to a plain `str` for the same reason:
        # `_represent_str` is registered for `str` only, so PyYAML would find no
        # representer for a subclass, e.g. `numpy.str_`.
        # A `UserString` is not a `str` at all, it is a `Sequence` whose items
        # are `UserString` instances, so the sequence branch below would recurse
        # until the stack is exhausted.
        return str(value)
    if isinstance(value, bytes):
        # Converted to plain `bytes` for the same reason: PyYAML represents the
        # binary scalar for `bytes` only, so a subclass such as `numpy.bytes_`
        # would find no representer. This branch precedes the `numpy.generic`
        # one below, which would otherwise handle that subclass.
        return bytes(value)
    if isinstance(value, Mapping):
        ancestor_ids.add(id(value))
        try:
            return {
                _convert_key_to_yaml_data(
                    key, ancestor_ids, array_store
                ): _convert_to_yaml_data(item, ancestor_ids, array_store)
                for key, item in value.items()
            }
        finally:
            ancestor_ids.discard(id(value))
    if isinstance(value, ndarray):
        # TODO: for big arrays, do not show all data? min, max, average?
        dtype = value.dtype
        kind = dtype.kind
        if (
            array_store is not None
            and value.size > _max_inline_array_size
            and kind != _object_dtype_kind
        ):
            return array_store.add_array(value)
        if kind in _representable_dtype_kinds or (
            kind == "f" and dtype.itemsize <= _max_unwrapped_float_itemsize
        ):
            return value.tolist()
        if kind == _stringifiable_dtype_kind:
            return value.astype(str).tolist()
        # An array of any other kind, e.g. object, timedelta, structured or
        # extended-precision floating point, yields items PyYAML cannot
        # represent, which would make the dump raise and lose the whole trace;
        # they are converted one by one, an object one possibly holding a
        # container of its own.
        ancestor_ids.add(id(value))
        try:
            return _convert_to_yaml_data(value.tolist(), ancestor_ids, array_store)
        finally:
            ancestor_ids.discard(id(value))
    if isinstance(value, generic):
        item = value.item()
        if isinstance(item, generic):
            # `numpy.longdouble.item` and `numpy.clongdouble.item` return a
            # NumPy scalar of the very same type, no Python type holding their
            # precision, so recursing on it would never terminate: the cycle
            # guard above cannot stop it either, `item` building a new object
            # at each round.
            return str(value)
        # Converted again, since the Python object a NumPy scalar unwraps to is
        # not necessarily representable either, e.g. a `numpy.complex128`
        # unwrapping to a `complex`, which reaches the `str` fallback below.
        return _convert_to_yaml_data(item, ancestor_ids, array_store)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, DesignSpace):
        # A design space is traced by the names of its variables, which is
        # what identifies the input space of a scenario; `str` would give
        # its pretty-printed table, from which nothing can be read back.
        return list(value.variables)
    if isinstance(value, BaseModel):
        # The fields are converted one by one rather than through
        # `model_dump`, so that a field holding a model is handled by this
        # branch as well, instead of being flattened into a mapping losing
        # the information below.
        ancestor_ids.add(id(value))
        try:
            data = {
                name: _convert_to_yaml_data(
                    getattr(value, name), ancestor_ids, array_store
                )
                for name in type(value).model_fields
            }
        finally:
            ancestor_ids.discard(id(value))
        if isinstance(value, BaseSettings):
            # `target_class_name` is a property backed by a class
            # variable, so it is not a model field, although it is what
            # identifies the object configured by the settings, e.g.
            # ``"MDF"`` for `MDF_Settings`.
            data["target_class_name"] = value.target_class_name
        return data
    if isinstance(value, Sequence | AbstractSet):
        # `AbstractSet`, and not `set`, since a `frozenset` is not a `set`; a
        # `tuple` needs no mention of its own, being already a `Sequence`.
        ancestor_ids.add(id(value))
        try:
            return [
                _convert_to_yaml_data(item, ancestor_ids, array_store) for item in value
            ]
        finally:
            ancestor_ids.discard(id(value))
    if value is None or type(value) in _plain_scalar_types:
        return value
    if isinstance(value, int):
        # A subclass of `int`, e.g. a user's own, normalized for the same
        # reason as a `str` subclass above; `bool` needs no branch of its own,
        # since it cannot be subclassed.
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, datetime):
        # A subclass of `datetime`, e.g. a `pandas.Timestamp`. Its ISO 8601
        # string is used rather than a rebuilt plain `datetime`, since a
        # subclass may carry a precision a `datetime` cannot hold, e.g. the
        # nanoseconds of a `pandas.Timestamp`.
        return value.isoformat()
    return str(value)


def _convert_key_to_yaml_data(
    key: Any, ancestor_ids: set[int], array_store: _NpyArrayStore | None
) -> Any:
    """Convert a mapping key into a representation writable by `dump_yaml`.

    A key goes through the same conversion as a value: PyYAML has no more a
    representer for a `Path`, an enum member or a NumPy scalar used as a key
    than used as a value, and an unconverted one would make the dump raise,
    hence break the observed call.

    A converted key that is no longer hashable, e.g. a tuple converted to a
    list, is replaced by the string representation of the original key: it
    could not be a key of the converted mapping at all. This also spares the
    reader of a trace the YAML complex key syntax (``? [1, 2]``), which not
    every YAML parser supports.

    Args:
        key: The key to convert.
        ancestor_ids: The identifiers of the containers being converted on the
            path from the root value down to this key's mapping.
        array_store: The store handling the large NumPy arrays met during the
            conversion, or `None` when every array must be converted inline.

    Returns:
        The converted key.
    """
    converted_key = _convert_to_yaml_data(key, ancestor_ids, array_store)
    try:
        hash(converted_key)
    except TypeError:
        return str(key)
    return converted_key


def load_yaml_trace(file_path: Path) -> Any:
    """Load a trace file, resolving every array reference into a NumPy array.

    A mapping matching exactly the `npy`, `dtype` and `shape` keys, see
    `_array_reference_keys`, is replaced by the NumPy array its `npy` entry
    points to, read with `numpy.load`. That entry is a path relative to the
    directory holding `file_path`, matching how `_NpyArrayStore` names it.

    Args:
        file_path: The path to the trace file.

    Returns:
        The trace data, with every array reference resolved into the NumPy
        array it points to.
    """
    with file_path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return _resolve_array_references(data, file_path.parent)


def _resolve_array_references(value: Any, directory_path: Path) -> Any:
    """Recursively replace array references with the arrays they point to.

    Args:
        value: The value to resolve.
        directory_path: The directory an array reference's `npy` entry is
            relative to.

    Returns:
        The resolved value.
    """
    if isinstance(value, Mapping):
        if set(value) == _array_reference_keys:
            return numpy.load(directory_path / value["npy"], allow_pickle=False)
        return {
            key: _resolve_array_references(item, directory_path)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_array_references(item, directory_path) for item in value]
    return value
