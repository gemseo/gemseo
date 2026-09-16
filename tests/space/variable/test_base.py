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
"""Tests for the base class of the variable hierarchy."""

from __future__ import annotations

import pickle
from copy import copy
from copy import deepcopy
from inspect import isabstract

import pytest
from numpy import array
from numpy import atleast_1d
from numpy import inf
from numpy import int32
from numpy import nan
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from gemseo.space.variable import BaseDeterministicVariable
from gemseo.space.variable import BaseIntervalVariable
from gemseo.space.variable import BaseNumericVariable
from gemseo.space.variable import BaseVariable
from gemseo.space.variable import ContinuousVariable
from gemseo.space.variable import DataType
from gemseo.space.variable import DiscreteVariable
from gemseo.space.variable import IntegerVariable
from gemseo.space.variable.random import RandomVariable
from gemseo.util.pydantic_ndarray import NDArrayPydantic  # noqa: TC001
from gemseo.util.testing.helper import assert_exception
from tests.space.variable.utils import kinds


def test_base_variable_is_abstract() -> None:
    """Check that the base variable cannot be instantiated."""
    assert isabstract(BaseVariable)
    assert BaseVariable.__abstractmethods__ == frozenset({"filter_components"})
    with pytest.raises(TypeError):
        BaseVariable()


@pytest.mark.parametrize(
    ("cls", "abstract_methods"),
    [
        (BaseNumericVariable, {"filter_components"}),
        (
            BaseDeterministicVariable,
            {
                "get_default_value",
                "get_normalization_mask",
                "filter_components",
            },
        ),
        (BaseIntervalVariable, {"get_normalization_mask"}),
    ],
)
def test_intermediate_classes_are_abstract(cls, abstract_methods) -> None:
    """Check what each layer of the hierarchy leaves to its subclasses."""
    assert isabstract(cls)
    assert cls.__abstractmethods__ == frozenset(abstract_methods)


@pytest.mark.parametrize(
    ("cls", "is_deterministic", "is_numeric"),
    [
        (ContinuousVariable, True, True),
        (IntegerVariable, True, True),
        (DiscreteVariable, True, True),
        (RandomVariable, False, True),
    ],
)
def test_axes_of_the_hierarchy(cls, is_deterministic, is_numeric) -> None:
    """Check the two axes along which a kind of variable is placed.

    A kind is either deterministic or random,
    and its components are numbers or not;
    only a kind whose components are numbers is bounded.
    """
    assert issubclass(cls, BaseVariable)
    assert issubclass(cls, BaseDeterministicVariable) is is_deterministic
    assert issubclass(cls, BaseNumericVariable) is is_numeric


@pytest.mark.parametrize(
    ("cls", "field_names"),
    [
        (ContinuousVariable, {"size", "lower_bound", "upper_bound"}),
        (IntegerVariable, {"size", "lower_bound", "upper_bound"}),
        (DiscreteVariable, {"choices"}),
        (RandomVariable, {"distribution_settings"}),
    ],
)
def test_fields_are_the_inputs(cls, field_names) -> None:
    """Check that the fields of a kind are exactly what the caller supplies.

    Whatever derives from these fields, e.g. the size of a scalar variable,
    is a read-only property, not a field.
    """
    assert set(cls.model_fields) == field_names


@pytest.mark.parametrize("cls", kinds)
@pytest.mark.parametrize(
    ("size", "lower_bound", "upper_bound"),
    [
        (1, 0, 0.0),
        (10, -5.0, 0.0),
        (1, -5.0, inf),
        (1, -inf, inf),
        (2, (-4, 4), inf),
    ],
)
def test_initialization(
    cls: type[BaseVariable],
    size: int,
    lower_bound: float | tuple[float, float],
    upper_bound: float,
) -> None:
    """Test the instantiation."""
    my_variable = cls(size=size, lower_bound=lower_bound, upper_bound=upper_bound)
    assert my_variable.size == size
    assert (my_variable.lower_bound == atleast_1d(lower_bound)).all()
    assert (my_variable.upper_bound == atleast_1d(upper_bound)).all()


@pytest.mark.parametrize("cls", kinds)
@pytest.mark.parametrize("size", [-1, 0])
def test_non_positive_size(cls, size, snapshot) -> None:
    """Check non-positive variables size."""
    with assert_exception(ValidationError, snapshot):
        cls(size=size)


@pytest.mark.parametrize("cls", kinds)
@pytest.mark.parametrize(
    "type_", ["complex", DataType.FLOAT, DataType.INTEGER, DataType.DISCRETE]
)
def test_type_is_not_settable(cls, type_, snapshot) -> None:
    """Check that the data type, pinned by the kind, cannot be passed.

    This holds whatever the value, including the data type pinned by the kind itself.
    """
    with assert_exception(ValidationError, snapshot):
        cls(type=type_)


@pytest.mark.parametrize("cls", kinds)
def test_unknown_field(cls, snapshot) -> None:
    """Check that an unknown field is rejected instead of being ignored."""
    with assert_exception(ValidationError, snapshot):
        cls(unknown=0)


@pytest.mark.parametrize("cls", kinds)
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_invalid_bound_size(cls, side, snapshot) -> None:
    """Check invalid bound size."""
    with assert_exception(ValidationError, snapshot):
        cls(**{f"{side}_bound": [0, 0]})


@pytest.mark.parametrize("cls", kinds)
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_invalid_bound_value_scalar(cls, side, snapshot) -> None:
    """Check invalid bound value type."""
    with assert_exception(ValidationError, snapshot):
        cls(**{f"{side}_bound": 1j})


@pytest.mark.parametrize("cls", kinds)
def test_wrong_boundaries(cls: type[BaseVariable], snapshot) -> None:
    """Test the instantiation with `upper_bound` lower than `lower_bound`."""
    with assert_exception(ValueError, snapshot):
        cls(size=1, lower_bound=0, upper_bound=-1)


@pytest.mark.parametrize("bound", ["lower_bound", "upper_bound"])
def test_frozen(variable, bound, snapshot) -> None:
    """Check that a variable is immutable (bounds cannot be reassigned)."""
    with assert_exception(ValidationError, snapshot):
        setattr(variable, bound, 0)


@pytest.mark.parametrize("cls", kinds)
def test_fields_are_inherited_unchanged(cls) -> None:
    """Check that a kind inherits the fields of the base interval variable as they are.

    The bounds are read through descriptors set on the base class
    once pydantic has built it,
    which pydantic must not mistake for the defaults of the fields of a subclass.
    """
    for name, field in BaseIntervalVariable.model_fields.items():
        assert cls.model_fields[name].default == field.default
        assert cls.model_fields[name].description == field.description

    assert not hasattr(cls, "lower_bound")


@pytest.mark.parametrize("cls", kinds)
@pytest.mark.parametrize("bound", ["lower_bound", "upper_bound"])
@pytest.mark.parametrize(
    ("attribute_name", "value"),
    [("shape", (2, 1)), ("strides", (0,)), ("dtype", int32)],
)
def test_bounds_are_handed_out_as_views(cls, bound, attribute_name, value) -> None:
    """Check that a bound is handed out as a view of what the variable stores.

    NumPy lets a caller reassign the shape, the strides and the data type
    of a read-only array, and these belong to the array object itself,
    so such a reassignment must reach the array of the caller only.
    """
    variable = cls(size=2, lower_bound=0, upper_bound=2)
    handed_out = getattr(variable, bound)
    assert handed_out is not getattr(variable, bound)

    setattr(handed_out, attribute_name, value)

    for name in ("lower_bound", "upper_bound"):
        assert getattr(variable, name).shape == (2,)
        assert getattr(variable, name).strides == (8,)
        assert getattr(variable, name).dtype == variable.component_type

    assert_array_equal(variable.get_default_value(), [1, 1])


@pytest.mark.parametrize("cls", kinds)
@pytest.mark.parametrize("bound", ["lower_bound", "upper_bound"])
def test_variable_built_from_a_bound_does_not_share_it(cls, bound) -> None:
    """Check that a variable built from the bound of another shares no array with it.

    A frozen bound is not copied when the variable freezes it,
    so the array stored must still not be the one the caller holds,
    which the caller can reshape.
    """
    variable = cls(size=2, lower_bound=0, upper_bound=2)
    supplied = getattr(variable, bound)
    other = cls(**{"size": 2, "lower_bound": 0, "upper_bound": 2, bound: supplied})

    supplied.shape = (2, 1)

    assert getattr(variable, bound).shape == (2,)
    assert getattr(other, bound).shape == (2,)
    assert_array_equal(getattr(other, bound), getattr(variable, bound))


@pytest.mark.parametrize("cls", kinds)
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("bound", [array([nan]), array([nan, nan])])
def test_bound_with_nan_components(cls, side, bound, snapshot) -> None:
    """Check a bound with one or several nan components."""
    with assert_exception(ValidationError, snapshot):
        cls(size=bound.size, **{f"{side}_bound": bound})


@pytest.mark.parametrize("cls", kinds)
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_multidimensional_bound(cls, side, snapshot) -> None:
    """Check a bound with more than one dimension."""
    with assert_exception(ValidationError, snapshot):
        cls(size=2, **{f"{side}_bound": array([[1.0, 2.0]])})


def test_model_copy_without_update(variable) -> None:
    """Check that copying a variable without an update returns the variable itself."""
    assert variable.model_copy() is variable
    assert variable.model_copy(deep=True) is variable


@pytest.mark.parametrize("cls", kinds)
def test_model_copy_with_inconsistent_update(cls, snapshot) -> None:
    """Check that an update inconsistent with the bounds is rejected.

    The base implementation of `model_copy` writes the update into `__dict__` without
    validating it.
    """
    variable = cls(size=2, lower_bound=0, upper_bound=1)
    with assert_exception(ValidationError, snapshot):
        variable.model_copy(update={"size": 5})


@pytest.mark.parametrize(
    ("cls", "type_"),
    [(ContinuousVariable, DataType.INTEGER), (IntegerVariable, DataType.FLOAT)],
)
def test_model_copy_with_another_type(cls, type_, snapshot) -> None:
    """Check that an update contradicting the pinned data type is rejected."""
    variable = cls(size=2, lower_bound=0, upper_bound=1)
    with assert_exception(ValidationError, snapshot):
        variable.model_copy(update={"type": type_})


def test_copy_and_pickle_keep_the_kind(variable, snapshot) -> None:
    """Check that copying and unpickling preserve the kind and the frozen bounds."""
    assert copy(variable) is variable
    assert deepcopy(variable) is variable

    restored = pickle.loads(pickle.dumps(variable))

    assert type(restored) is type(variable)
    assert restored == variable
    assert not restored.lower_bound.flags.writeable
    assert not restored.upper_bound.flags.writeable

    # The restored bounds have been refrozen, so they cannot be thawed.
    with assert_exception(ValueError, snapshot):
        restored.lower_bound.setflags(write=True)

    with assert_exception(ValueError, snapshot):
        restored.upper_bound.setflags(write=True)


@pytest.mark.parametrize("enable_integer_normalization", [False, True])
@pytest.mark.parametrize("upper_bound", [1, inf])
@pytest.mark.parametrize("cls", kinds)
def test_get_normalization_mask(cls, upper_bound, enable_integer_normalization) -> None:
    """Check the per-component normalization policy of a variable."""
    variable = cls(size=2, lower_bound=0, upper_bound=upper_bound)
    policy = variable.get_normalization_mask(enable_integer_normalization)
    expected = upper_bound != inf and (
        cls is ContinuousVariable or enable_integer_normalization
    )
    assert_array_equal(policy, [expected] * 2)


@pytest.mark.parametrize("cls", kinds)
def test_find_components_outside_domain_with_none_and_inf(cls) -> None:
    """Check that None and infinite components are in the domain of a variable."""
    variable = cls(size=2, lower_bound=0, upper_bound=inf)
    value = array([None, inf], dtype=object)
    assert variable.find_components_outside_domain(value) == set()


def test_eq_is_data_based() -> None:
    """Check that equality compares the data and not the exact class."""
    continuous = ContinuousVariable(size=2, lower_bound=0, upper_bound=1)
    integer = IntegerVariable(size=2, lower_bound=0, upper_bound=1)

    assert continuous == ContinuousVariable(size=2, lower_bound=0, upper_bound=1)
    assert continuous != integer
    assert continuous != ContinuousVariable(size=2, lower_bound=0, upper_bound=2)
    assert continuous != "not a variable"
    # The sizes differ, so the comparison returns before reaching the bounds.
    assert continuous != ContinuousVariable(size=3, lower_bound=0, upper_bound=1)


def test_eq_compares_the_fields_of_a_subclass() -> None:
    """Check that a field added by a subclass takes part in the comparison."""

    class CustomVariable(ContinuousVariable):
        """A continuous variable with extra fields."""

        label: str = ""
        weights: NDArrayPydantic[float] = array([0.0])

    variable = CustomVariable(size=1, lower_bound=0, upper_bound=1, label="a")

    assert variable == CustomVariable(size=1, lower_bound=0, upper_bound=1, label="a")
    assert variable != CustomVariable(size=1, lower_bound=0, upper_bound=1, label="b")
    # An array field whose components cannot be compared element-wise.
    assert variable != CustomVariable(
        size=1, lower_bound=0, upper_bound=1, label="a", weights=array([0.0, 0.0])
    )
    # The extra fields are declared by only one of the two kinds.
    assert variable != ContinuousVariable(size=1, lower_bound=0, upper_bound=1)
