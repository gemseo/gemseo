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
from typing import TYPE_CHECKING
from typing import Any

import pytest
from numpy import array
from numpy import ndarray
from numpy.testing import assert_array_equal

from gemseo import set_data_converters
from gemseo.core.data_converters.base import BaseDataConverter
from gemseo.core.grammars.json import JSONGrammar
from gemseo.core.grammars.pydantic import PydanticGrammar
from gemseo.core.grammars.simple import SimpleGrammar
from gemseo.utils.comparisons import compare_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Generator


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


VALUE_TO_ARRAY_DATA = [
    ("a_str", "0", array(["0"])),
    ("a_float", 0.0, array([0.0])),
    ("a_int", 0, array([0])),
    ("a_complex", 1j, array([1j])),
    ("a_ndarray", array([0.0]), array([0.0])),
]


@pytest.mark.parametrize(("name", "value", "expected"), VALUE_TO_ARRAY_DATA)
def test_convert_value_to_array(converter, name, value, expected) -> None:
    """Verify convert_value_to_array."""
    assert_array_equal(converter.convert_value_to_array(name, value), expected)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        *VALUE_TO_ARRAY_DATA,
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


ARRAY_TO_VALUE_DATA = [
    ("a_float", array([0.0]), 0.0),
    ("a_int", array([0]), 0),
    ("a_complex", array([1j]), 1j),
    ("a_ndarray", array([0.0]), array([0.0])),
]


@pytest.mark.parametrize(("name", "value", "expected"), ARRAY_TO_VALUE_DATA)
def test_convert_array_to_value(converter, name, value, expected) -> None:
    """Verify convert_array_to_value."""
    assert converter.convert_array_to_value(name, value) == expected


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        *ARRAY_TO_VALUE_DATA,
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


VALUE_SIZE_DATA = [
    ("a_str", "0", 1),
    ("a_float", 0.0, 1),
    ("a_int", 0, 1),
    ("a_complex", 1j, 1),
    ("a_ndarray", array([0.0] * 2), 2),
]


@pytest.mark.parametrize(("name", "value", "expected"), VALUE_SIZE_DATA)
def test_get_value_size(converter, name, value, expected) -> None:
    """Verify get_value_size."""
    assert converter.get_value_size(name, value) == expected


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        *VALUE_SIZE_DATA,
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


DATA = {
    "a_float": 0.0,
    "a_int": 0,
    "a_complex": 1j,
    "a_ndarray": array([0.0] * 2),
    "a_str": "0",
}
NAMES_TO_SLICES = {
    "a_float": slice(0, 1, None),
    "a_int": slice(1, 2, None),
    "a_complex": slice(2, 3, None),
    "a_ndarray": slice(3, 5, None),
    "a_str": slice(5, 6, None),
}
ARRAY = array([0.0, 0, 1.0j, 0.0, 0.0, "0"], dtype=object)


def test_compute_name_to_slices(converter) -> None:
    """Verify compute_name_to_slices."""
    # Without name_to_size.
    name_to_slice, end = converter.compute_name_to_slice(DATA.keys(), DATA)

    assert name_to_slice == NAMES_TO_SLICES
    assert end == 6

    # Without name_to_size.
    name_to_size = {
        "a_int": 1,
        "a_ndarray": 2,
    }
    name_to_slice, end = converter.compute_name_to_slice(
        DATA.keys(), DATA, name_to_size=name_to_size
    )

    assert name_to_slice == NAMES_TO_SLICES
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

    name_to_size = converter.compute_name_to_size(DATA.keys(), DATA)

    assert name_to_size == expected


def test_convert_array_to_data(converter) -> None:
    """Verify convert_array_to_data."""
    data = converter.convert_array_to_data(ARRAY, NAMES_TO_SLICES)
    assert compare_dict_of_arrays(data, DATA)


def test_convert_data_to_array(converter) -> None:
    """Verify convert_data_to_array."""
    # Full data.
    array_ = converter.convert_data_to_array(DATA.keys(), DATA)
    # Cast the reference to the same dtype because numpy.concatenate
    # does too to the dtype that can represent all the data.
    ref_array = ARRAY.astype(array_.dtype)
    assert_array_equal(array_, ref_array)
    # Non full data.
    array_ = converter.convert_data_to_array(("a_int", "a_ndarray"), DATA)
    assert_array_equal(array_, array([0.0] * 3))
    # No data.
    array_ = converter.convert_data_to_array((), DATA)
    assert len(array_) == 0


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
