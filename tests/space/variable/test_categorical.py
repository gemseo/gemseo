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
"""Tests for the categorical variable."""

from __future__ import annotations

import pickle

import pytest
from numpy import array
from numpy import int64
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from gemseo.space.variable import BaseNumericVariable
from gemseo.space.variable import CategoricalVariable
from gemseo.space.variable import DataType
from gemseo.space.variable import DiscreteVariable
from gemseo.util.testing.helper import assert_exception


def test_fields() -> None:
    """Check the fields of a categorical variable."""
    variable = CategoricalVariable(categories=["steel", "aluminium", "titanium"])

    assert variable.type == DataType.CATEGORICAL
    assert variable.size == 1
    assert variable.coordinate_type is int64
    # The declaration order is kept.
    assert variable.categories == ("steel", "aluminium", "titanium")
    # The variable has neither bounds nor cast.
    assert not isinstance(variable, BaseNumericVariable)
    assert not hasattr(variable, "lower_bound")
    assert not hasattr(variable, "upper_bound")
    assert not hasattr(variable, "cast")


@pytest.mark.parametrize("categories", [["a", "b"], ("a", "b")])
def test_categories_types(categories) -> None:
    """Check that the categories can be given as a list or a tuple of strings."""
    assert CategoricalVariable(categories=categories).categories == ("a", "b")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"categories": []},
        {"categories": ()},
        {"categories": ["a", "a"]},
        {"categories": ["a", "b", "a", "b"]},
        {"categories": ["a", 1]},
        {"categories": [1, 2]},
        {"categories": "ab"},
        {"categories": [None]},
        {"categories": ["a"], "size": 2},
        {"categories": ["a"], "lower_bound": 0},
        {},
    ],
)
def test_rejections(kwargs, snapshot) -> None:
    """Check the inputs rejected by a categorical variable."""
    with assert_exception(ValidationError, snapshot):
        CategoricalVariable(**kwargs)


@pytest.mark.parametrize("enable_integer_normalization", [False, True])
def test_normalization_mask(enable_integer_normalization) -> None:
    """Check that a categorical variable is normalized like an integer one."""
    variable = CategoricalVariable(categories=["a", "b"])
    assert_array_equal(
        variable.get_normalization_mask(enable_integer_normalization),
        [enable_integer_normalization],
    )


def test_normalization_mask_is_read_only(snapshot) -> None:
    """Check that the normalization mask shared by the categorical variables is frozen."""  # noqa: E501
    variable = CategoricalVariable(categories=["a", "b"])
    mask = variable.get_normalization_mask(True)
    assert not mask.flags.writeable
    with assert_exception(ValueError, snapshot):
        mask[0] = False

    mask.shape = (1, 1)
    assert variable.get_normalization_mask(True).shape == (1,)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, set()),
        (2, set()),
        (2.0, set()),
        (None, set()),
        (3, {0}),
        (-1, {0}),
        (0.5, {0}),
        (float("nan"), {0}),
    ],
)
def test_find_components_outside_domain(value, expected) -> None:
    """Check that only the position of a category is in the domain."""
    variable = CategoricalVariable(categories=["a", "b", "c"])
    dtype = object if value is None else float
    assert variable.find_components_outside_domain(array([value], dtype=dtype)) == (
        expected
    )


def test_default_value() -> None:
    """Check that the default value is the position of the first category."""
    variable = CategoricalVariable(categories=["b", "a"])
    assert_array_equal(variable.get_default_value(), [0])
    assert variable.get_default_value().dtype == int64


def test_filter_components() -> None:
    """Check that a categorical variable keeping its only component is an identity."""
    variable = CategoricalVariable(categories=["a", "b"])
    assert variable.filter_components([0]) is variable


@pytest.mark.parametrize("components", [[], [0, 0], [1], [0, 1]])
def test_filter_components_rejects_other_selections(components, snapshot) -> None:
    """Check that a categorical variable can only keep its single component."""
    variable = CategoricalVariable(categories=["a", "b"])
    with assert_exception(ValueError, snapshot):
        variable.filter_components(components)


def test_encode_and_decode() -> None:
    """Check the conversion between labels and coordinates."""
    variable = CategoricalVariable(categories=["steel", "aluminium", "titanium"])

    positions = variable.encode(["titanium", "steel", "titanium"])
    assert_array_equal(positions, [2, 0, 2])
    assert positions.dtype == int64
    assert_array_equal(variable.encode("aluminium"), [1])

    labels = variable.decode(positions)
    assert_array_equal(labels, ["titanium", "steel", "titanium"])
    # The shape of the coordinates is kept.
    assert variable.decode(array([[1.0], [0.0]])).shape == (2, 1)


@pytest.mark.parametrize("labels", [["gold"], ["steel", "gold"], [1], [None]])
def test_encode_unknown_label(labels, snapshot) -> None:
    """Check that a label that is not a category cannot be encoded."""
    variable = CategoricalVariable(categories=["steel", "aluminium"])
    with assert_exception(ValueError, snapshot):
        variable.encode(labels)


@pytest.mark.parametrize(
    "positions", [[2], [-1], [0, 5], [0.5], [float("nan")], [[0, 1], [2, 0.5]]]
)
def test_decode_unknown_position(positions, snapshot) -> None:
    """Check that a coordinate that is not a position cannot be decoded."""
    variable = CategoricalVariable(categories=["steel", "aluminium"])
    with assert_exception(ValueError, snapshot):
        variable.decode(array(positions))


def test_out_of_domain_messages(snapshot) -> None:
    """Check the wording of an out-of-domain coordinate."""
    variable = CategoricalVariable(categories=["a", "b"])
    assert variable._get_out_of_domain_message("x", array([3]), {0}) == snapshot
    assert variable._get_out_of_domain_component_message("x", 0, 3) == snapshot


def test_eq() -> None:
    """Check that the categories and their order take part in the comparison."""
    variable = CategoricalVariable(categories=["a", "b"])

    assert variable == CategoricalVariable(categories=("a", "b"))
    assert variable != CategoricalVariable(categories=["b", "a"])
    assert variable != CategoricalVariable(categories=["a", "b", "c"])
    assert variable != DiscreteVariable(choices=[0, 1])


def test_copy_and_pickle() -> None:
    """Check that copying and unpickling preserve the variable."""
    variable = CategoricalVariable(categories=["a", "b"])
    restored = pickle.loads(pickle.dumps(variable))

    assert type(restored) is CategoricalVariable
    assert restored == variable
    assert restored.categories == ("a", "b")


def test_model_copy() -> None:
    """Check that a copy can update the categories."""
    variable = CategoricalVariable(categories=["a", "b"])
    new_variable = variable.model_copy(update={"categories": ["c", "d", "e"]})
    assert new_variable.categories == ("c", "d", "e")
    assert variable.categories == ("a", "b")
