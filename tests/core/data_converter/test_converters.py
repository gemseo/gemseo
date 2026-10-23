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
from __future__ import annotations

import operator
from enum import StrEnum
from pathlib import Path
from pathlib import PurePath
from pathlib import PurePosixPath
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Any
from typing import Literal

import pytest
from numpy import array
from numpy import ndarray
from numpy.testing import assert_array_equal
from pydantic import DirectoryPath
from pydantic import FilePath
from pydantic import NewPath

from gemseo import set_data_converters
from gemseo.core.data_converter.base import BaseDataConverter
from gemseo.core.grammar.json import JSONGrammar
from gemseo.core.grammar.pydantic import PydanticGrammar
from gemseo.core.grammar.simple import SimpleGrammar
from gemseo.util.comparison import compare_dict_of_arrays
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Generator


class _StrEnum(StrEnum):
    """A string enumeration."""

    A = "a"


def create_converters() -> list[BaseDataConverter]:
    """Create all the converter types."""
    name_to_type = {
        "a_bool": bool,
        "a_float": float,
        "a_int": int,
        "a_complex": complex,
        "a_ndarray": ndarray,
        "a_str": str,
    }

    converters = []

    for cls in (SimpleGrammar, JSONGrammar, PydanticGrammar):
        grammar = cls("g")
        grammar.update_from_types(name_to_type)
        converters.append(grammar.data_converter)

    return converters


@pytest.fixture(params=create_converters())
def converter(request) -> BaseDataConverter:
    """Return a converter."""
    return request.param


@pytest.fixture
def reset_converters() -> Generator[None, Any, None]:
    """Fixture that reset the user converters."""
    yield
    BaseDataConverter.value_to_array_converters.clear()
    BaseDataConverter.array_to_value_converters.clear()
    BaseDataConverter.value_size_getters.clear()


def test_is_numeric(converter) -> None:
    """Verify is_numeric."""
    for name in ("a_bool",):
        assert not converter.is_numeric(name)
    for name in ("a_float", "a_int", "a_complex", "a_ndarray"):
        assert converter.is_numeric(name)


def test_can_differentiate(converter) -> None:
    """Verify is_continuous."""
    for name in ("a_bool", "a_int"):
        assert not converter.is_continuous(name)
    for name in ("a_float", "a_complex", "a_ndarray"):
        assert converter.is_continuous(name)


value_to_array_data = [
    ("a_str", "0", array(["0"])),
    ("a_float", 0.0, array([0.0])),
    ("a_int", 0, array([0])),
    ("a_complex", 1j, array([1j])),
    ("a_ndarray", array([0.0]), array([0.0])),
]


@pytest.mark.parametrize(("name", "value", "expected"), value_to_array_data)
def test_convert_value_to_array(converter, name, value, expected) -> None:
    """Verify convert_value_to_array."""
    assert_array_equal(converter.convert_value_to_array(name, value), expected)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        *value_to_array_data,
        ("a_dict", {"dummy": 1j}, array([1j])),
        ("a_2d_ndarray", array([[0.0]]), array([0.0])),
    ],
)
def test_convert_value_to_array_user(
    reset_converters, converter, name, value, expected
) -> None:
    """Verify convert_value_to_array with user converters."""
    set_data_converters(
        {
            "a_dict": operator.itemgetter("dummy"),
            "a_2d_ndarray": operator.itemgetter(0),
        },
        {},
        {},
    )
    assert_array_equal(converter.convert_value_to_array(name, value), expected)


array_to_value_data = [
    ("a_float", array([0.0]), 0.0),
    ("a_int", array([0]), 0),
    ("a_complex", array([1j]), 1j),
    ("a_ndarray", array([0.0]), array([0.0])),
    # The item of an object array is not a NumPy scalar.
    ("a_str", array(["0"], dtype=object), "0"),
]


@pytest.mark.parametrize(("name", "value", "expected"), array_to_value_data)
def test_convert_array_to_value(converter, name, value, expected) -> None:
    """Verify convert_array_to_value."""
    value_ = converter.convert_array_to_value(name, value)
    assert value_ == expected
    # NumPy scalars shall have been converted to Python scalars.
    assert type(value_) is type(expected)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        *array_to_value_data,
        ("a_dict", array([1j]), {"dummy": 1j}),
        ("a_2d_ndarray", array([0.0]), array([[0.0]])),
    ],
)
def test_convert_array_to_value_user(
    reset_converters, converter, name, value, expected
) -> None:
    """Verify convert_array_to_value with user converters."""
    set_data_converters(
        {},
        {
            "a_dict": lambda a: {"dummy": a},
            "a_2d_ndarray": lambda a: array([a]),
        },
        {},
    )
    assert converter.convert_array_to_value(name, value) == expected


value_size_data = [
    ("a_str", "0", 1),
    ("a_path", Path("0"), 1),
    ("a_float", 0.0, 1),
    ("a_int", 0, 1),
    ("a_complex", 1j, 1),
    ("a_ndarray", array([0.0] * 2), 2),
]


@pytest.mark.parametrize(("name", "value", "expected"), value_size_data)
def test_get_value_size(converter, name, value, expected) -> None:
    """Verify get_value_size."""
    assert converter.get_value_size(name, value) == expected


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        *value_size_data,
        ("a_dict", {"dummy": array(1)}, 1),
    ],
)
def test_get_value_sizes_user(
    reset_converters, converter, name, value, expected
) -> None:
    """Verify get_value_size with user converters."""
    set_data_converters(
        {},
        {},
        {
            "a_dict": lambda v: v["dummy"].size,
        },
    )
    assert converter.get_value_size(name, value) == expected


def test_get_value_size_non_numeric(converter, snapshot) -> None:
    """Verify get_value_size raises a TypeError for a non-numeric, non-array value."""
    with assert_exception(TypeError, snapshot):
        converter.get_value_size("a_list", [1.0])


def test_convert_data_to_array_paths_and_strs() -> None:
    """Verify that paths and strings are concatenated as strings."""
    grammar = PydanticGrammar("g")
    grammar.update_from_types({"a": Path, "b": str})
    converter = grammar.data_converter
    data = {"a": Path("x"), "b": "y"}
    assert_array_equal(converter.convert_data_to_array(["a", "b"], data), ["x", "y"])


reference_data = {
    "a_float": 0.0,
    "a_int": 0,
    "a_complex": 1j,
    "a_ndarray": array([0.0] * 2),
    "a_str": "0",
}
names_to_slices = {
    "a_float": slice(0, 1, None),
    "a_int": slice(1, 2, None),
    "a_complex": slice(2, 3, None),
    "a_ndarray": slice(3, 5, None),
    "a_str": slice(5, 6, None),
}
reference_array = array([0.0, 0, 1.0j, 0.0, 0.0, "0"], dtype=object)


def test_compute_name_to_slices(converter) -> None:
    """Verify compute_name_to_slices."""
    # Without name_to_size.
    name_to_slice, end = converter.compute_name_to_slice(
        reference_data.keys(), reference_data
    )

    assert name_to_slice == names_to_slices
    assert end == 6

    # Without name_to_size.
    name_to_size = {
        "a_int": 1,
        "a_ndarray": 2,
    }
    name_to_slice, end = converter.compute_name_to_slice(
        reference_data.keys(), reference_data, name_to_size=name_to_size
    )

    assert name_to_slice == names_to_slices
    assert end == 6


def test_compute_name_to_sizes(converter) -> None:
    """Verify compute_name_to_sizes."""
    expected = {
        "a_float": 1,
        "a_int": 1,
        "a_complex": 1,
        "a_ndarray": 2,
        "a_str": 1,
    }

    name_to_size = converter.compute_name_to_size(reference_data.keys(), reference_data)

    assert name_to_size == expected


def test_convert_array_to_data(converter) -> None:
    """Verify convert_array_to_data."""
    data = converter.convert_array_to_data(reference_array, names_to_slices)
    assert compare_dict_of_arrays(data, reference_data)


def test_convert_data_to_array(converter) -> None:
    """Verify convert_data_to_array."""
    # Full numeric data.
    numeric_names = [name for name in reference_data if name != "a_str"]
    array_ = converter.convert_data_to_array(numeric_names, reference_data)
    assert_array_equal(array_, array([0.0, 0, 1.0j, 0.0, 0.0]))
    # Non full data.
    array_ = converter.convert_data_to_array(("a_int", "a_ndarray"), reference_data)
    assert_array_equal(array_, array([0.0] * 3))
    # No data.
    array_ = converter.convert_data_to_array((), reference_data)
    assert len(array_) == 0


def test_convert_data_to_array_mixed_types(converter, snapshot) -> None:
    """Verify that convert_data_to_array refuses strings mixed with numbers."""
    with assert_exception(TypeError, snapshot):
        converter.convert_data_to_array(reference_data.keys(), reference_data)


def create_converters_ndarray() -> list[BaseDataConverter]:
    """Create converter for checking ndarray dtype features."""
    names_to_data = {
        "float_array": array([1.0]),
        "int_array": array([1]),
        "complex_array": array([1j]),
        "str_array": array(["hello"]),
    }

    converters = []

    for cls in (JSONGrammar, PydanticGrammar):
        grammar = cls("g")
        grammar.update_from_data(names_to_data)
        converters.append(grammar.data_converter)

    return converters


@pytest.fixture(params=create_converters_ndarray())
def converter_ndarray(request) -> BaseDataConverter:
    """Return a converter for checking ndarray."""
    return request.param


def test_is_numeric_typed_numeric_arrays(converter_ndarray) -> None:
    """Numeric-dtype arrays are numeric."""
    for name in ("float_array", "int_array", "complex_array"):
        assert converter_ndarray.is_numeric(name)

    assert not converter_ndarray.is_numeric("str_array")


def test_is_continuous_typed_arrays(converter_ndarray) -> None:
    """Float/complex-dtype arrays are continuous; int/str-dtype arrays are not."""
    assert converter_ndarray.is_continuous("float_array")
    assert converter_ndarray.is_continuous("complex_array")
    assert not converter_ndarray.is_continuous("int_array")
    assert not converter_ndarray.is_continuous("str_array")


def test_pydantic_to_simple_grammar_normalizes_ndarray_type() -> None:
    """Verify PydanticGrammar.to_simple_grammar normalizes _NDArrayPydantic to ndarray.

    When an MDA (SimplerGrammar) absorbs types from PydanticGrammar disciplines,
    _NDArrayPydantic must be normalized to ndarray so that the SimpleGrammar data
    converter correctly identifies numeric and continuous variables.
    """
    pydantic_grammar = PydanticGrammar("g")
    pydantic_grammar.update_from_data({
        "x": array([1.0]),
        "a_file": "my_file",
        "a_int": 1,
    })
    simple_grammar = pydantic_grammar.to_simple_grammar()
    converter = simple_grammar.data_converter

    assert simple_grammar["x"] is ndarray
    assert converter.is_numeric("x")
    assert converter.is_continuous("x")
    assert not converter.is_numeric("a_file")
    assert converter.is_numeric("a_int")
    assert not converter.is_continuous("a_int")


@pytest.mark.parametrize(
    ("grammar_class", "declared_type", "expected_type"),
    [
        (SimpleGrammar, Path, Path),
        (SimpleGrammar, PurePath, PurePath),
        (SimpleGrammar, PurePosixPath, PurePosixPath),
        (SimpleGrammar, PureWindowsPath, PureWindowsPath),
        (PydanticGrammar, Path, Path),
        (PydanticGrammar, PurePosixPath, PurePosixPath),
        (PydanticGrammar, PureWindowsPath, PureWindowsPath),
        (PydanticGrammar, Path | None, Path),
        (PydanticGrammar, PureWindowsPath | None, PureWindowsPath),
        (PydanticGrammar, PurePosixPath | PureWindowsPath, Path),
        (PydanticGrammar, FilePath, Path),
        (PydanticGrammar, NewPath, Path),
        (PydanticGrammar, DirectoryPath | None, Path),
        (PydanticGrammar, Annotated[PureWindowsPath | None, "foo"], PureWindowsPath),
        (
            PydanticGrammar,
            Annotated[PureWindowsPath, "x"] | PureWindowsPath,
            PureWindowsPath,
        ),
        (PydanticGrammar, Annotated[PurePosixPath, ["foo"]], PurePosixPath),
        (
            PydanticGrammar,
            PurePosixPath | Annotated[PureWindowsPath, {"k": 1}],
            Path,
        ),
    ],
    ids=[
        "simple-path",
        "simple-pure-path",
        "simple-posix",
        "simple-windows",
        "pydantic-path",
        "pydantic-posix",
        "pydantic-windows",
        "optional-path",
        "optional-windows",
        "posix-or-windows",
        "file-path",
        "new-path",
        "optional-directory-path",
        "annotated-optional",
        "path-declared-twice",
        "unhashable-metadata",
        "union-with-unhashable-metadata",
    ],
)
def test_path_round_trip(grammar_class, declared_type, expected_type) -> None:
    """Verify that a path is converted to a string array and back to a path."""
    grammar = grammar_class("g")
    grammar.update_from_types({"p": declared_type})
    converter = grammar.data_converter
    path = PurePosixPath("some") / "path"
    array_ = converter.convert_value_to_array("p", path)
    assert array_.dtype.kind == "U"
    assert array_[0] == "some/path"
    value = converter.convert_array_to_value("p", array_)
    assert isinstance(value, expected_type)
    assert value.as_posix() == "some/path"


def test_path_to_array_platform() -> None:
    """Verify that a path is stored with forward slashes whatever the platform."""
    grammar = SimpleGrammar("g")
    grammar.update_from_types({"p": PureWindowsPath})
    array_ = grammar.data_converter.convert_value_to_array("p", PureWindowsPath("a\\b"))
    assert array_[0] == "a/b"


@pytest.mark.parametrize(
    ("grammar_class", "declared_type"),
    [
        (SimpleGrammar, str),
        (JSONGrammar, str),
        (PydanticGrammar, str),
        (PydanticGrammar, str | Path),
        (PydanticGrammar, Path | str | None),
        (PydanticGrammar, FilePath | str),
        (PydanticGrammar, Literal["a"] | Path),
        (PydanticGrammar, _StrEnum | Path),
        (PydanticGrammar, int | Path),
        (PydanticGrammar, Path | list[Annotated[str, {"k": 1}]]),
    ],
    ids=[
        "simple-str",
        "json-str",
        "pydantic-str",
        "str-or-path",
        "optional-path-or-str",
        "file-path-or-str",
        "literal-or-path",
        "enum-or-path",
        "int-or-path",
        "unhashable-metadata",
    ],
)
def test_str_array_to_value_without_path(grammar_class, declared_type) -> None:
    """Verify that a string array is converted to a str for an item not only a path."""
    grammar = grammar_class("g")
    grammar.update_from_types({"s": declared_type})
    value = grammar.data_converter.convert_array_to_value("s", array(["a"]))
    assert value == "a"
    assert type(value) is str


@pytest.mark.parametrize("declared_type", [None, ndarray], ids=["no-type", "ndarray"])
def test_str_array_to_value_without_path_type(declared_type) -> None:
    """Verify that a one-item string array is unchanged without path type."""
    grammar = SimpleGrammar("g")
    grammar.update_from_types({"x": declared_type})
    array_ = array(["red"])
    assert grammar.data_converter.convert_array_to_value("x", array_) is array_


def test_path_to_value_without_type() -> None:
    """Verify that a path of a SimpleGrammar item without type is read as an array."""
    grammar = SimpleGrammar("g")
    grammar.update_from_types({"p": None})
    converter = grammar.data_converter
    array_ = converter.convert_value_to_array("p", PurePosixPath("a") / "b")
    value = converter.convert_array_to_value("p", array_)
    assert isinstance(value, ndarray)
    assert_array_equal(value, array(["a/b"]))


@pytest.mark.parametrize(
    ("grammar_class", "declared_type", "expected_type"),
    [
        (SimpleGrammar, Path, Path),
        (SimpleGrammar, PureWindowsPath, PureWindowsPath),
        (PydanticGrammar, Path, Path),
        (PydanticGrammar, Path | None, Path),
    ],
    ids=[
        "simple-path",
        "simple-windows",
        "pydantic-path",
        "optional-path",
    ],
)
def test_bytes_array_to_path(grammar_class, declared_type, expected_type) -> None:
    """Verify that a one-item bytes array is converted to a path."""
    grammar = grammar_class("g")
    grammar.update_from_types({"p": declared_type})
    value = grammar.data_converter.convert_array_to_value("p", array([b"a/b"]))
    assert isinstance(value, expected_type)
    assert value.as_posix() == "a/b"


@pytest.mark.parametrize(
    ("declared_type", "expected_type", "expected"),
    [
        (str | Path, str, "a/b"),
        (Literal["a"] | Path, str, "a/b"),
        (_StrEnum | Path, str, "a/b"),
        (int | Path, str, "a/b"),
        (str, bytes, b"a/b"),
    ],
    ids=[
        "str-or-path",
        "literal-or-path",
        "enum-or-path",
        "int-or-path",
        "str-only",
    ],
)
def test_bytes_array_to_str_with_path(declared_type, expected_type, expected) -> None:
    """Verify that a one-item bytes array is decoded to a str only if a path type is
    declared, and is unchanged otherwise.
    """
    grammar = PydanticGrammar("g")
    grammar.update_from_types({"b": declared_type})
    value = grammar.data_converter.convert_array_to_value("b", array([b"a/b"]))
    assert type(value) is expected_type
    assert value == expected


@pytest.mark.parametrize("declared_type", [None, Path], ids=["no-type", "path"])
def test_str_array_of_several_items_to_value(declared_type) -> None:
    """Verify that a string array of several items is not converted to a str."""
    grammar = SimpleGrammar("g")
    grammar.update_from_types({"s": declared_type})
    array_ = array(["a", "b"])
    assert_array_equal(
        grammar.data_converter.convert_array_to_value("s", array_), array_
    )


def test_path_type_after_grammar_update() -> None:
    """Verify that the path type follows a change of the grammar."""
    grammar = PydanticGrammar("g")
    grammar.update_from_types({"p": Path})
    converter = grammar.data_converter
    assert isinstance(converter.convert_array_to_value("p", array(["a"])), Path)
    grammar.update_from_types({"p": PureWindowsPath})
    value = converter.convert_array_to_value("p", array(["a"]))
    assert isinstance(value, PureWindowsPath)
