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
"""Tests for the variable hierarchy."""

from __future__ import annotations

import pickle
from copy import copy
from copy import deepcopy
from inspect import isabstract

import pytest
from numpy import array
from numpy import atleast_1d
from numpy import float64
from numpy import inf
from numpy import int64
from numpy import nan
from numpy.testing import assert_array_equal
from pydantic import ValidationError

from gemseo.space._variable import BaseVariable
from gemseo.space._variable import ContinuousVariable
from gemseo.space._variable import DataType
from gemseo.space._variable import IntegerVariable
from gemseo.space._variable import Variable
from gemseo.space._variable import VariableFactory
from gemseo.util.pydantic_ndarray import NDArrayPydantic  # noqa: TC001
from gemseo.util.testing.helper import assert_exception

KINDS = (ContinuousVariable, IntegerVariable)


@pytest.fixture(params=KINDS)
def variable(request) -> BaseVariable:
    """A variable of each kind, of size 1 and with the bounds [0, 1]."""
    return request.param(size=1, lower_bound=0, upper_bound=1)


def test_base_variable_is_abstract() -> None:
    """Check that the base variable cannot be instantiated."""
    assert isabstract(BaseVariable)
    assert BaseVariable.__abstractmethods__ == frozenset({"compute_normalization_mask"})
    with pytest.raises(TypeError):
        BaseVariable()


@pytest.mark.parametrize(
    ("cls", "type_"),
    [(ContinuousVariable, DataType.FLOAT), (IntegerVariable, DataType.INTEGER)],
)
def test_init_defaults(cls, type_) -> None:
    """Test the default values of __init__."""
    v = cls()
    assert v.type == type_
    assert v.size == 1
    assert v.lower_bound == -array([inf])
    assert v.upper_bound == array([inf])


@pytest.mark.parametrize("cls", KINDS)
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


@pytest.mark.parametrize("cls", KINDS)
@pytest.mark.parametrize("size", [-1, 0])
def test_non_positive_size(cls, size, snapshot) -> None:
    """Check non-positive variables size."""
    with assert_exception(ValidationError, snapshot):
        cls(size=size)


@pytest.mark.parametrize("cls", KINDS)
def test_invalid_type(cls, snapshot) -> None:
    """Check invalid variable type."""
    with assert_exception(ValidationError, snapshot):
        cls(type="complex")


@pytest.mark.parametrize(
    ("cls", "type_"),
    [(ContinuousVariable, DataType.INTEGER), (IntegerVariable, DataType.FLOAT)],
)
def test_type_of_another_kind(cls, type_, snapshot) -> None:
    """Check that a kind rejects the data type pinned by another kind."""
    with assert_exception(ValidationError, snapshot):
        cls(type=type_)


@pytest.mark.parametrize("cls", KINDS)
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_invalid_bound_size(cls, side, snapshot) -> None:
    """Check invalid bound size."""
    with assert_exception(ValidationError, snapshot):
        cls(**{f"{side}_bound": [0, 0]})


@pytest.mark.parametrize("cls", KINDS)
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_invalid_bound_value_scalar(cls, side, snapshot) -> None:
    """Check invalid bound value type."""
    with assert_exception(ValidationError, snapshot):
        cls(**{f"{side}_bound": 1j})


@pytest.mark.parametrize("cls", KINDS)
def test_wrong_boundaries(cls: type[BaseVariable], snapshot) -> None:
    """Test the instantiation with `upper_bound` lower than `lower_bound`."""
    with assert_exception(ValueError, snapshot):
        cls(size=1, lower_bound=0, upper_bound=-1)


@pytest.mark.parametrize("bound", ["lower_bound", "upper_bound"])
def test_frozen(variable, bound, snapshot) -> None:
    """Check that a variable is immutable (bounds cannot be reassigned)."""
    with assert_exception(ValidationError, snapshot):
        setattr(variable, bound, 0)


@pytest.mark.parametrize("cls", KINDS)
@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("bound", [array([nan]), array([nan, nan])])
def test_bound_with_nan_components(cls, side, bound, snapshot) -> None:
    """Check a bound with one or several nan components."""
    with assert_exception(ValidationError, snapshot):
        cls(size=bound.size, **{f"{side}_bound": bound})


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("bound", [array([1.5]), array([1.5, 2.5])])
def test_bound_with_non_integer_components(side, bound, snapshot) -> None:
    """Check a bound with one or several non-integer components."""
    with assert_exception(ValidationError, snapshot):
        IntegerVariable(size=bound.size, **{f"{side}_bound": bound})


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("bound", [array([1.5]), array([1.5, 2.5])])
def test_continuous_bound_with_non_integer_components(side, bound) -> None:
    """Check that a continuous variable accepts non-integer bound components."""
    kwargs = {"lower_bound": -inf, "upper_bound": inf, f"{side}_bound": bound}
    my_variable = ContinuousVariable(size=bound.size, **kwargs)
    assert_array_equal(getattr(my_variable, f"{side}_bound"), bound)


@pytest.mark.parametrize("cls", KINDS)
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_multidimensional_bound(cls, side, snapshot) -> None:
    """Check a bound with more than one dimension."""
    with assert_exception(ValidationError, snapshot):
        cls(size=2, **{f"{side}_bound": array([[1.0, 2.0]])})


def test_model_copy_without_update(variable) -> None:
    """Check that copying a variable without an update returns the variable itself."""
    assert variable.model_copy() is variable
    assert variable.model_copy(deep=True) is variable


@pytest.mark.parametrize("cls", KINDS)
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


def test_model_copy_leaves_original_alone() -> None:
    """Check that an update returns a new variable and does not touch the original."""
    variable = ContinuousVariable(size=2, lower_bound=0.0, upper_bound=1.0)

    new_variable = variable.model_copy(update={"lower_bound": array([-9.0, -9.0])})

    assert new_variable is not variable
    assert_array_equal(new_variable.lower_bound, array([-9.0, -9.0]))
    assert not new_variable.lower_bound.flags.writeable
    # The base implementation would have written the update into the original.
    assert_array_equal(variable.lower_bound, array([0.0, 0.0]))
    assert not variable.lower_bound.flags.writeable


def test_model_copy_converts_the_update() -> None:
    """Check that a scalar bound of an update is converted and typed as expected."""
    variable = IntegerVariable(size=2, lower_bound=0, upper_bound=10)

    new_variable = variable.model_copy(update={"upper_bound": 3})

    assert isinstance(new_variable, IntegerVariable)
    assert new_variable.type == DataType.INTEGER
    assert new_variable.upper_bound.dtype == int64
    assert_array_equal(new_variable.upper_bound, array([3, 3]))


def test_copy_and_pickle_keep_the_kind(variable) -> None:
    """Check that copying and unpickling preserve the kind and the frozen bounds."""
    assert copy(variable) is variable
    assert deepcopy(variable) is variable

    restored = pickle.loads(pickle.dumps(variable))

    assert type(restored) is type(variable)
    assert restored == variable
    assert not restored.lower_bound.flags.writeable
    assert not restored.upper_bound.flags.writeable


@pytest.mark.parametrize(
    ("cls", "expected"), [(ContinuousVariable, float64), (IntegerVariable, int64)]
)
def test_component_type(cls, expected) -> None:
    """Check the NumPy type of the components of a variable."""
    assert cls().component_type is expected


def test_cast_continuous() -> None:
    """Check that a continuous variable casts to float but preserves a complex value."""
    variable = ContinuousVariable(size=2)
    assert variable.cast(array([1, 2])).dtype == float64
    complex_value = array([1.0 + 1.0j, 2.0 + 2.0j])
    cast = variable.cast(complex_value)
    assert cast.dtype == complex_value.dtype
    # The value is copied, so that the caller does not keep a hand on it.
    assert cast is not complex_value
    assert_array_equal(cast, complex_value)


def test_cast_integer() -> None:
    """Check that an integer variable casts to int."""
    cast = IntegerVariable(size=2).cast(array([1.6, 2.6]))
    assert cast.dtype == int64
    assert_array_equal(cast, array([1, 2]))


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected"),
    [(-inf, inf, 0.0), (-inf, 2.0, 2.0), (1.0, inf, 1.0), (1.0, 3.0, 2.0)],
)
def test_compute_default_component(lower_bound, upper_bound, expected) -> None:
    """Check the default value of a component."""
    assert (
        ContinuousVariable.compute_default_component(lower_bound, upper_bound)
        == expected
    )


@pytest.mark.parametrize("enable_integer_normalization", [False, True])
@pytest.mark.parametrize("upper_bound", [1, inf])
@pytest.mark.parametrize("cls", KINDS)
def test_compute_normalization_mask(
    cls, upper_bound, enable_integer_normalization
) -> None:
    """Check the per-component normalization policy of a variable."""
    variable = cls(size=2, lower_bound=0, upper_bound=upper_bound)
    policy = variable.compute_normalization_mask(enable_integer_normalization)
    expected = upper_bound != inf and (
        cls is ContinuousVariable or enable_integer_normalization
    )
    assert_array_equal(policy, [expected] * 2)


@pytest.mark.parametrize(
    ("cls", "expected"), [(ContinuousVariable, set()), (IntegerVariable, {1})]
)
def test_find_components_outside_domain(cls, expected) -> None:
    """Check the components outside the domain of a variable."""
    variable = cls(size=2, lower_bound=0, upper_bound=10)
    assert variable.find_components_outside_domain(array([1.0, 1.5])) == expected


@pytest.mark.parametrize("cls", KINDS)
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


@pytest.mark.parametrize("kind", KINDS)
def test_unpickle_legacy_variable(kind) -> None:
    """Check that a variable pickled before the hierarchy is restored as its kind."""
    legacy = Variable(
        size=2,
        type=kind.model_fields["type"].default,
        lower_bound=array([0.0, 0.0]),
        upper_bound=array([1.0, 2.0]),
    )

    with pytest.warns(DeprecationWarning, match=kind.__name__):
        restored = pickle.loads(pickle.dumps(legacy))

    assert type(restored) is kind
    assert restored == kind(
        size=2, lower_bound=array([0.0, 0.0]), upper_bound=array([1.0, 2.0])
    )
    # The restored variable has been validated as a new one:
    # its bounds are frozen so that they cannot be mutated in place.
    assert not restored.lower_bound.flags.writeable
    assert not restored.upper_bound.flags.writeable


def test_unpickle_legacy_variable_with_default_fields() -> None:
    """Check that a legacy variable with no field is restored with the defaults."""
    with pytest.warns(
        DeprecationWarning,
        match="The class 'gemseo.space._variable.Variable' is deprecated",
    ):
        restored = pickle.loads(pickle.dumps(Variable()))

    assert restored == ContinuousVariable(size=1, lower_bound=-inf, upper_bound=inf)


def test_legacy_variable_is_not_a_kind() -> None:
    """Check that the legacy variable is out of the hierarchy and of its factory.

    Otherwise the factory would find two classes pinning the float data type.
    """
    assert not issubclass(Variable, BaseVariable)
    assert "Variable" not in VariableFactory().class_names
