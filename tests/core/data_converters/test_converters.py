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

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import ndarray
from numpy.testing import assert_array_equal

from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.pydantic_grammar import PydanticGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.utils.comparisons import compare_dict_of_arrays

if TYPE_CHECKING:
    from gemseo.core.data_converters.base import BaseDataConverter


def create_converters() -> list[BaseDataConverter]:
    """Create all the converter types."""
    names_to_types = {
        "a_bool": bool,
        "a_float": float,
        "a_int": int,
        "a_complex": complex,
        "a_ndarray": ndarray,
    }

    converters = []

    for cls in (SimpleGrammar, JSONGrammar, PydanticGrammar):
        grammar = cls("g")
        grammar.update_from_types(names_to_types)
        converters.append(grammar.data_converter)

    return converters


@pytest.fixture(params=create_converters())
def converter(request) -> BaseDataConverter:
    """Return a converter."""
    return request.param


def test_is_numeric(converter):
    """Verify is_numeric."""
    for name in ("a_bool",):
        assert not converter.is_numeric(name)
    for name in ("a_float", "a_int", "a_complex", "a_ndarray"):
        assert converter.is_numeric(name)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("a_float", 0.0, array([0.0])),
        ("a_int", 0, array([0])),
        ("a_complex", 1j, array([1j])),
        ("a_ndarray", array([0.0]), array([0.0])),
    ],
)
def test_convert_value_to_array(converter, name, value, expected):
    """Verify convert_value_to_array."""
    assert_array_equal(converter.convert_value_to_array(name, value), expected)


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("a_float", array([0.0]), 0.0),
        ("a_int", array([0]), 0),
        ("a_complex", array([1j]), 1j),
        ("a_ndarray", array([0.0]), array([0.0])),
    ],
)
def test_convert_array_to_value(converter, name, value, expected):
    """Verify convert_array_to_value."""
    assert converter.convert_array_to_value(name, value) == expected


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("a_float", 0.0, 1),
        ("a_int", 0, 1),
        ("a_complex", 1j, 1),
        ("a_ndarray", array([0.0] * 2), 2),
    ],
)
def test_get_value_size(converter, name, value, expected):
    """Verify get_value_size."""
    assert converter.get_value_size(name, value) == expected


DATA = {
    "a_float": 0.0,
    "a_int": 0,
    "a_complex": 1j,
    "a_ndarray": array([0.0] * 2),
}
NAMES_TO_SLICES = {
    "a_float": slice(0, 1, None),
    "a_int": slice(1, 2, None),
    "a_complex": slice(2, 3, None),
    "a_ndarray": slice(3, 5, None),
}
ARRAY = array([0.0, 0.0, 1.0j, 0.0, 0.0])


def test_compute_name_to_slices(converter):
    """Verify compute_name_to_slices."""
    # Without names_to_sizes.
    names_to_slices, end = converter.compute_names_to_slices(DATA.keys(), DATA)

    assert names_to_slices == NAMES_TO_SLICES
    assert end == 5

    # Without names_to_sizes.
    names_to_sizes = {
        "a_int": 1,
        "a_ndarray": 2,
    }
    names_to_slices, end = converter.compute_names_to_slices(
        DATA.keys(), DATA, names_to_sizes=names_to_sizes
    )

    assert names_to_slices == NAMES_TO_SLICES
    assert end == 5


def test_compute_name_to_sizes(converter):
    """Verify compute_name_to_sizes."""
    expected = {
        "a_float": 1,
        "a_int": 1,
        "a_complex": 1,
        "a_ndarray": 2,
    }

    names_to_sizes = converter.compute_names_to_sizes(DATA.keys(), DATA)

    assert names_to_sizes == expected


def test_convert_array_to_data(converter):
    """Verify convert_array_to_data."""
    data = converter.convert_array_to_data(ARRAY, NAMES_TO_SLICES)
    assert compare_dict_of_arrays(data, DATA)


def test_convert_data_to_array(converter):
    """Verify convert_data_to_array."""
    # Full data.
    array_ = converter.convert_data_to_array(DATA.keys(), DATA)
    assert_array_equal(array_, ARRAY)
    # Non full data.
    array_ = converter.convert_data_to_array(("a_int", "a_ndarray"), DATA)
    assert_array_equal(array_, array([0.0] * 3))
    # No data.
    array_ = converter.convert_data_to_array((), DATA)
    assert len(array_) == 0
