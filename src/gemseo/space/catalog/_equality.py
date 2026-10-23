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
"""The equality of the properties of two catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from itertools import starmap
from typing import Any

from numpy import array_equal
from numpy import ndarray
from pandas import Series

from gemseo.space.catalog._input import is_missing


def properties_are_equal(
    property_values: ndarray, other_property_values: ndarray
) -> bool:
    """Return whether two properties hold the same values.

    A missing value equals a missing value,
    so that a catalog carrying one equals itself.
    `array_equal` handles that through `equal_nan`,
    but only for a pair of float, complex, datetime or timedelta properties:
    it raises on any other pair, e.g. a pair of string properties,
    so a property of Python objects, e.g. one mixing numbers and `None`,
    is compared element by element instead.

    Args:
        property_values: The property.
        other_property_values: The property to compare it with.

    Returns:
        Whether the two properties hold the same values.
    """
    kinds = {property_values.dtype.kind, other_property_values.dtype.kind}
    if "O" not in kinds:
        return array_equal(
            property_values,
            other_property_values,
            equal_nan=kinds <= {"f", "c", "M", "m"},
        )

    # Missing, not blank: an empty string is a value here, and must not be
    # read as equal to a None that happens to share its blankness.
    return all(
        starmap(
            _values_are_equal, zip(property_values, other_property_values, strict=True)
        )
    )


def _values_are_equal(value: Any, other_value: Any) -> bool:
    """Return whether two values of an object property are equal.

    A missing value equals a missing value, and never anything else: e.g. a
    pandas `pd.NA` compared with a value that is not missing is itself
    missing, not `False`, and `bool` raises on it rather than reading it as a
    falsy comparison, so neither side may reach a bare `==` unless both are
    known not to be missing.
    Two arrays, two pandas `Series`, two mappings or two sequences
    are compared structurally, before any bare `==`,
    so that a missing value nested in them equals a missing value too,
    whatever the nesting.

    Args:
        value: The value.
        other_value: The value to compare it with.

    Returns:
        Whether the two values are equal.
    """
    if value is other_value:
        return True

    if is_missing(value) or is_missing(other_value):
        return is_missing(value) and is_missing(other_value)

    are_equal = _nested_values_are_equal(value, other_value)
    if are_equal is not None:
        return are_equal

    try:
        return bool(value == other_value)
    except ValueError:
        # The truth value of an array of several elements is ambiguous,
        # e.g. for an array compared with a value that is not an array.
        return False


def _nested_values_are_equal(value: Any, other_value: Any) -> bool | None:
    """Return whether two containers of an object property are equal.

    The arrays, the pandas `Series`, the mappings and the sequences
    are compared element-wise,
    each element with
    [_values_are_equal][gemseo.space.catalog._equality._values_are_equal];
    two `Series` must also have equal indices,
    and two sequences the same type.
    A string or a byte string is not compared as a sequence.

    Args:
        value: The value.
        other_value: The value to compare it with.

    Returns:
        Whether the two values are equal,
        `None` if they are not two containers of the same kind.
    """
    if isinstance(value, Series) and isinstance(other_value, Series):
        return value.index.equals(other_value.index) and _nested_values_are_equal(
            value.to_numpy(), other_value.to_numpy()
        )

    if isinstance(value, ndarray) and isinstance(other_value, ndarray):
        return value.shape == other_value.shape and all(
            starmap(_values_are_equal, zip(value.flat, other_value.flat, strict=True))
        )

    if isinstance(value, Mapping) and isinstance(other_value, Mapping):
        return value.keys() == other_value.keys() and all(
            _values_are_equal(item, other_value[key]) for key, item in value.items()
        )

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and type(value) is type(other_value)
    ):
        return len(value) == len(other_value) and all(
            starmap(_values_are_equal, zip(value, other_value, strict=True))
        )

    return None
