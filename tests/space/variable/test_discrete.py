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
"""Tests for the discrete variable."""

from __future__ import annotations

import pickle

import pytest
from numpy import array
from numpy import float64
from numpy import inf
from numpy import nan
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from gemseo.space.variable import ContinuousVariable
from gemseo.space.variable import DataType
from gemseo.space.variable import DiscreteVariable
from gemseo.util.testing.helper import assert_exception


def test_fields() -> None:
    """Check the fields of a discrete variable."""
    variable = DiscreteVariable(choices=[0.72, 0.45, 0.55])

    assert variable.type == DataType.DISCRETE
    assert variable.size == 1
    assert variable.component_type is float64
    # The choices are sorted at construction.
    assert_array_equal(variable.choices, array([0.45, 0.55, 0.72]))
    assert variable.choices.dtype == float64
    # The bounds are derived from the sorted choices.
    assert_array_equal(variable.lower_bound, array([0.45]))
    assert_array_equal(variable.upper_bound, array([0.72]))


@pytest.mark.parametrize("choices", [array([1, 4, 6, 9]), [1, 4, 6, 9], (1, 4, 6, 9)])
def test_choices_types(choices) -> None:
    """Check that the choices can be given as an array, a list or a tuple."""
    variable = DiscreteVariable(choices=choices)
    assert_array_equal(variable.choices, array([1.0, 4.0, 6.0, 9.0]))


def test_choices_are_isolated() -> None:
    """Check that the caller does not keep a hand on the choices."""
    choices = array([3.0, 1.0])
    variable = DiscreteVariable(choices=choices)
    choices[0] = 99.0
    assert_array_equal(variable.choices, array([1.0, 3.0]))


def test_choices_are_frozen() -> None:
    """Check that the choices cannot be mutated in place."""
    variable = DiscreteVariable(choices=[1, 2])
    assert not variable.choices.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        variable.choices[0] = 3.0


def test_choices_are_deduplicated() -> None:
    """Check that duplicated choices are silently dropped, and the rest sorted."""
    variable = DiscreteVariable(choices=[2.0, 4.0, 2.0, 1.0, 4.0])
    assert_array_equal(variable.choices, array([1.0, 2.0, 4.0]))


def test_with_a_single_choice() -> None:
    """Check a discrete variable with a single choice."""
    variable = DiscreteVariable(choices=[42])
    assert_array_equal(variable.lower_bound, variable.upper_bound)
    assert_array_equal(variable.compute_default_value(), array([42.0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"choices": []},
        {"choices": array([])},
        {"choices": [nan]},
        {"choices": [inf, 1]},
        {"choices": array([[1.0, 2.0], [3.0, 4.0]])},
        {"choices": [1, 2], "lower_bound": 0},
        {"choices": [1, 2], "upper_bound": 3},
        {"choices": [1, 2], "size": 2},
        {"choices": ["a", "b"]},
        {"choices": [None]},
        {"choices": [nan, nan, 1.0]},
    ],
)
def test_rejections(kwargs, snapshot) -> None:
    """Check the inputs rejected by a discrete variable."""
    with assert_exception(ValidationError, snapshot):
        DiscreteVariable(**kwargs)


def test_normalization_mask() -> None:
    """Check that a discrete variable is never normalized."""
    variable = DiscreteVariable(choices=[1, 2])
    assert_array_equal(variable.compute_normalization_mask(False), array([False]))
    assert_array_equal(variable.compute_normalization_mask(True), array([False]))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.45, set()), (0.72, set()), (0.6, {0}), (0.0, {0}), (None, set())],
)
def test_find_components_outside_domain(value, expected) -> None:
    """Check that only a choice is in the domain of a discrete variable."""
    variable = DiscreteVariable(choices=[0.45, 0.72])
    dtype = object if value is None else float64
    assert variable.find_components_outside_domain(array([value], dtype=dtype)) == (
        expected
    )


def test_default_value() -> None:
    """Check that the default value of a discrete variable is its first value."""
    variable = DiscreteVariable(choices=[6, 2, 4])
    # Not the center 4.0 of the derived bounds.
    assert_array_equal(variable.compute_default_value(), array([2.0]))
    assert variable.compute_default_value().dtype == float64


@pytest.mark.parametrize(
    ("choices", "expected"),
    [
        ([1, 2], "[1.0, 2.0]"),
        (list(range(6)), "[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]"),
        (list(range(7)), "[0.0, 1.0, 2.0, ..., 4.0, 5.0, 6.0] (7 choices)"),
        (
            list(range(2, 100, 2)),
            "[2.0, 4.0, 6.0, ..., 94.0, 96.0, 98.0] (49 choices)",
        ),
    ],
)
def test_format_choices(choices, expected) -> None:
    """Check the rendering of the choices, elided beyond six values."""
    variable = DiscreteVariable(choices=choices)
    assert variable._format_choices() == expected


def test_eq() -> None:
    """Check that the choices take part in the comparison."""
    variable = DiscreteVariable(choices=[1, 2, 3])

    assert variable == DiscreteVariable(choices=[3, 2, 1])
    # The two variables share their derived bounds but not their choices.
    assert variable != DiscreteVariable(choices=[1, 3])
    assert variable != ContinuousVariable(size=1, lower_bound=1, upper_bound=3)


def test_pickle_freezes_the_choices() -> None:
    """Check that unpickling refreezes the choices."""
    variable = DiscreteVariable(choices=[1, 2])
    restored = pickle.loads(pickle.dumps(variable))

    assert type(restored) is DiscreteVariable
    assert restored == variable
    assert not restored.choices.flags.writeable


@pytest.mark.parametrize("bound", ["lower_bound", "upper_bound"])
def test_model_copy_rejects_the_bounds(bound, snapshot) -> None:
    """Check that a copy cannot re-derive the bounds of a discrete variable.

    This must hold even though the bounds derived from the choices
    always sit in `__dict__`, and so are always carried over by `model_copy`,
    whether or not the caller's update names them.
    """
    variable = DiscreteVariable(choices=[1, 2])
    with assert_exception(ValidationError, snapshot):
        variable.model_copy(update={bound: 0})


def test_model_copy_accepts_an_unrelated_update() -> None:
    """Check that a copy not touching the bounds re-derives them without error.

    Regression test: the bounds derived from the choices always sit in
    `__dict__`, so a naive re-validation of the whole `__dict__` would wrongly
    trip the "bounds are not settable" check for any update, even one that
    leaves the bounds alone.
    """
    variable = DiscreteVariable(choices=[1, 2])

    new_variable = variable.model_copy(update={"choices": [3, 4]})

    assert_array_equal(new_variable.choices, array([3.0, 4.0]))
    assert_array_equal(new_variable.lower_bound, array([3.0]))
    assert_array_equal(new_variable.upper_bound, array([4.0]))
    # The original is left alone.
    assert_array_equal(variable.choices, array([1.0, 2.0]))


def test_format_out_of_domain_values() -> None:
    """Check the wording of an out-of-domain value, always for a single component.

    A discrete variable is scalar, so `find_components_outside_domain` can only
    ever return a single index; the plural wording of the message is dead code.
    """
    variable = DiscreteVariable(choices=[1, 2])
    message = variable._get_out_of_domain_message("x", array([3.0]), {0})
    assert message == (
        "The following value of variable 'x' is not among its choices "
        "[1.0, 2.0]: 3.0 (index 0)."
    )
