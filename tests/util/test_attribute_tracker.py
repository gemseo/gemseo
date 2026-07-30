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
"""Tests for the attributes tracker."""

from __future__ import annotations

import pytest
from numpy import asarray
from numpy import float64
from numpy import multiply
from numpy import ones
from numpy import zeros
from numpy.testing import assert_allclose
from pydantic import BaseModel
from pydantic import Field
from pydantic import PrivateAttr
from pydantic import ValidationError
from pydantic import computed_field

from gemseo.util.attributes_tracker import _create_tracker_class
from gemseo.util.attributes_tracker import _ModelTrackerMixin
from gemseo.util.attributes_tracker import _NDArrayTracker
from gemseo.util.attributes_tracker import _SequenceTracker
from gemseo.util.attributes_tracker import track_model
from gemseo.util.attributes_tracker import wrap_with_attributes_tracking
from gemseo.util.pydantic_ndarray import NDArrayPydantic


class Sub(BaseModel):
    no: int = 0
    r: int
    rw: int
    w: int
    r_sequence_item: list[int] = [0]
    rw_sequence_item: list[int] = [0]
    w_sequence_item: list[int] = [0]
    r_sequence: list[int] = [0]
    rw_sequence: list[int] = [0]
    w_sequence: list[int] = [0]

    def helper(self) -> int:
        return 7


class SubWO(BaseModel):
    w: int = 0


class Main(BaseModel):
    no: int = 0

    sub: Sub = Sub(r=0, rw=0, w=0)

    subwo: SubWO = SubWO()

    r_int: int
    rw_int: int
    w_int: int

    r_str: str = "0"
    rw_str: str = "0"
    w_str: str = "0"

    r_bool: bool = False
    rw_bool: bool = False
    w_bool: bool = False

    r_array: NDArrayPydantic[float64] = zeros(1)
    rw_array: NDArrayPydantic[float64] = zeros(1)
    w_array: NDArrayPydantic[float64] = zeros(1)

    r_array_2d_1: NDArrayPydantic[float64] = zeros((1, 1))
    rw_array_2d_1: NDArrayPydantic[float64] = zeros((1, 1))
    w_array_2d_1: NDArrayPydantic[float64] = zeros((1, 1))

    r_array_2d_2: NDArrayPydantic[float64] = zeros((1, 1))
    rw_array_2d_2: NDArrayPydantic[float64] = zeros((1, 1))
    w_array_2d_2: NDArrayPydantic[float64] = zeros((1, 1))

    def helper(self) -> int:
        return 42

    @computed_field
    @property
    def derived(self) -> int:
        return self.r_int + self.rw_int


@pytest.fixture
def tracker():
    return wrap_with_attributes_tracking(Main(r_int=0, rw_int=0, w_int=0))


def f(b):
    assert b.sub.r == 0
    assert b.sub.rw == 0
    b.sub.rw = 1
    b.sub.w = 1

    assert b.sub.r_sequence == [0]
    assert b.sub.rw_sequence == [0]
    b.sub.rw_sequence = [1]
    b.sub.w_sequence = [1]

    assert b.sub.r_sequence_item[0] == 0
    assert b.sub.rw_sequence_item[0] == 0
    b.sub.rw_sequence_item[0] = 1
    b.sub.w_sequence_item[0] = 1

    b.subwo.w = 1

    assert b.r_int == 0
    assert b.rw_int == 0
    b.rw_int = 1
    b.w_int = 1

    assert b.r_str == "0"
    assert b.rw_str == "0"
    b.rw_str = "1"
    b.w_str = "1"

    assert not b.r_bool
    assert not b.rw_bool
    b.rw_bool = True
    b.w_bool = True

    assert b.r_array == zeros(1)
    assert b.rw_array == zeros(1)
    b.rw_array = ones(1)
    b.w_array = ones(1)

    assert b.r_array_2d_1 == zeros((1, 1))
    assert b.rw_array_2d_1[0, 0] == 0
    b.rw_array_2d_1[0, 0] = 1
    b.w_array_2d_1[0, 0] = 1

    assert b.r_array_2d_2[0, ...] == [0.0]
    assert b.rw_array_2d_2[0, ...] == [0.0]
    b.rw_array_2d_2[0, ...] = 1
    b.w_array_2d_2[0, ...] = 1


def test_tracker():
    b = Main(r_int=0, rw_int=0, w_int=0)
    b = wrap_with_attributes_tracking(b)
    f(b)

    input_model = b.get_input_model()
    assert input_model.model_fields.keys() == {
        "r_int",
        "rw_int",
        "r_str",
        "rw_str",
        "r_bool",
        "rw_bool",
        "r_array",
        "rw_array",
        "r_array_2d_1",
        "rw_array_2d_1",
        "r_array_2d_2",
        "rw_array_2d_2",
        "sub.r",
        "sub.rw",
        "sub.r_sequence",
        "sub.rw_sequence",
        "sub.r_sequence_item",
        "sub.rw_sequence_item",
    }

    output_model = b.get_output_model()
    assert output_model.model_fields.keys() == {
        "w_int",
        "rw_int",
        "w_str",
        "rw_str",
        "w_bool",
        "rw_bool",
        "w_array",
        "rw_array",
        "w_array_2d_1",
        "rw_array_2d_1",
        "w_array_2d_2",
        "rw_array_2d_2",
        "sub.w",
        "sub.rw_sequence",
        "sub.w_sequence",
        "sub.rw_sequence_item",
        "sub.w_sequence_item",
        "sub.rw",
        "subwo.w",
    }

    # Those assert must come last to avoid toggling tracking.
    assert b.sub.w == 1
    assert b.sub.rw == 1
    assert b.subwo.w == 1
    assert b.sub.w_sequence_item[0] == 1
    assert b.sub.rw_sequence_item[0] == 1
    assert b.rw_int == 1
    assert b.w_int == 1
    assert b.rw_str == "1"
    assert b.w_str == "1"
    assert b.rw_bool
    assert b.w_bool
    assert b.rw_array == ones(1)
    assert b.w_array == ones(1)
    assert b.rw_array_2d_1 == ones((1, 1))
    assert b.w_array_2d_1 == ones((1, 1))
    assert b.rw_array_2d_2 == ones((1, 1))
    assert b.w_array_2d_2 == ones((1, 1))

    assert b.rw_array_2d_1[0][0] == 1
    assert b.rw_array_2d_1[0] == ones(1)


def test_wrap_non_model_raises() -> None:
    """Tracking a non-model object must raise a clear error."""
    with pytest.raises(TypeError, match="shall be a Pydantic BaseModel"):
        wrap_with_attributes_tracking([1, 2, 3])
    with pytest.raises(TypeError, match="shall be a Pydantic BaseModel"):
        track_model(42)


def test_method_call_not_tracked_as_read(tracker) -> None:
    """Calling a model method must not register it as a read field."""
    assert tracker.helper() == 42
    assert "helper" not in tracker.attrs_read
    assert "helper" not in tracker.get_input_model().model_fields


def test_computed_field_dependencies_tracked(tracker) -> None:
    """Reading a computed_field must register the fields it reads instead."""
    assert tracker.derived == 0
    assert "derived" not in tracker.attrs_read
    assert tracker.get_input_model().model_fields.keys() == {"r_int", "rw_int"}


def test_computed_field_setattr_raises(tracker) -> None:
    """A computed_field cannot be written through the tracker."""
    # The wording differs across Python versions: "has no setter" (>=3.11)
    # versus "can't set attribute" (3.10).
    with pytest.raises(AttributeError, match=r"has no setter|can't set attribute"):
        tracker.derived = 1


def test_setattr_non_field_raises(tracker) -> None:
    """Setting an attribute that is not a model field must raise."""
    with pytest.raises(ValueError, match="object has no field"):
        tracker.new_attr = 1
    assert "new_attr" not in tracker.attrs_written


class DefaultFactoryModel(BaseModel):
    xs: list[float] = Field(default_factory=lambda: [1.0, 2.0])
    y: float = 0.0


def test_default_factory_field() -> None:
    """Check that a field declared with a default factory can be tracked."""
    tracker = wrap_with_attributes_tracking(DefaultFactoryModel())
    tracker.y = tracker.xs[0] + 1.0
    input_model = tracker.get_input_model()
    assert input_model.model_fields.keys() == {"xs"}
    assert input_model.model_fields["xs"].default == [1.0, 2.0]
    assert tracker.get_output_model().model_fields.keys() == {"y"}


class ImmutableSequencesModel(BaseModel):
    t: tuple[int, ...] = (1, 2, 3)
    b: bytes = b"\x01\x02"


def test_immutable_sequence_fields() -> None:
    """Check that immutable sequence fields can be read and tracked."""
    tracker = wrap_with_attributes_tracking(ImmutableSequencesModel())
    first, second, third = tracker.t
    assert (first, second, third) == (1, 2, 3)
    assert len(tracker.t) == 3
    assert tracker.t[0] == 1
    assert tracker.b[1] == 2
    assert tracker.get_input_model().model_fields.keys() == {"t", "b"}


def test_nested_method_not_tracked(tracker) -> None:
    """Method access on a nested submodel must not appear in read fields."""
    assert tracker.sub.helper() == 7
    assert "sub" in tracker.attrs_read
    assert "helper" not in tracker.sub.attrs_read


class MethodModel(BaseModel):
    a: float = 1.0
    b: float = 2.0
    out: float = 0.0

    _cache: float = PrivateAttr(default=0.0)

    def get_total(self) -> float:
        return self.a + self.b

    def update_out(self) -> None:
        self.out = self.get_total()

    def update_cache(self) -> None:
        self._cache = self.a


def test_method_field_accesses_tracked() -> None:
    """Fields read and written inside model methods must be tracked."""
    tracker = wrap_with_attributes_tracking(MethodModel())
    tracker.update_out()
    assert tracker.get_input_model().model_fields.keys() == {"a", "b"}
    assert tracker.get_output_model().model_fields.keys() == {"out"}


def test_tracked_model_isinstance(tracker) -> None:
    """The tracked model must be an instance of the model class."""
    assert isinstance(tracker, Main)


def test_original_model_not_modified() -> None:
    """The tracked model is a copy: the original model must not change."""
    model = MethodModel()
    tracker = wrap_with_attributes_tracking(model)
    tracker.update_out()
    tracker.a = 9.0
    assert model.a == 1.0
    assert model.out == 0.0


def test_private_attribute_assignment() -> None:
    """Private attributes must be assignable and not tracked."""
    tracker = wrap_with_attributes_tracking(MethodModel())
    tracker.update_cache()
    assert tracker._cache == 1.0
    assert tracker.get_input_model().model_fields.keys() == {"a"}
    assert not tracker.get_output_model().model_fields


class ArrayModel(BaseModel):
    x: NDArrayPydantic[float64] = ones(2)
    y: NDArrayPydantic[float64] = zeros(2)
    m: NDArrayPydantic[float64] = zeros((2, 2))


def test_ndarray_expression_results_are_plain_arrays() -> None:
    """Arithmetic on tracked arrays must yield usable plain arrays."""
    tracker = wrap_with_attributes_tracking(ArrayModel())
    tmp = tracker.x * 2.0
    assert tmp[0] == 2.0
    tracker.y = tmp + 1.0
    assert tracker.get_input_model().model_fields.keys() == {"x"}
    assert tracker.get_output_model().model_fields.keys() == {"y"}


def test_ndarray_inplace_writes_tracked() -> None:
    """In-place array mutations must be recorded as writes."""
    tracker = wrap_with_attributes_tracking(ArrayModel())
    tracker.y += 1.0
    tracker.m.fill(7.0)
    multiply(tracker.x, 2.0, out=tracker.y)
    assert tracker.get_output_model().model_fields.keys() == {"y", "m"}
    assert_allclose(asarray(tracker.y), 2.0 * ones(2))
    assert_allclose(asarray(tracker.m), 7.0 * ones((2, 2)))


def test_ndarray_aggregate_read_classification() -> None:
    """A field read via aggregation and written elementwise is input and output."""
    tracker = wrap_with_attributes_tracking(ArrayModel())
    total = float(tracker.x.sum())
    tracker.x[0] = total
    assert "x" in tracker.get_input_model().model_fields
    assert "x" in tracker.get_output_model().model_fields


def test_ndarray_views_tracked() -> None:
    """Writes through row and slice views must be recorded."""
    tracker = wrap_with_attributes_tracking(ArrayModel())
    tracker.m[0][0] = 5.0
    tracker.m[1:2][0, 1] = 9.0
    row_total = 0.0
    for row in tracker.m:
        row_total += row[0]
    assert row_total == 5.0
    assert tracker.m[0, 0] == 5.0
    assert tracker.m[1, 1] == 9.0
    assert "m" in tracker.get_output_model().model_fields
    assert "m" in tracker.get_input_model().model_fields


class Inner(BaseModel):
    a: float = 1.0
    b: float = 2.0


class Outer(BaseModel):
    x: float = 1.0
    inner: Inner = Inner()


def test_whole_submodel_assignment_flattened() -> None:
    """Assigning a whole sub-model must expose its fields as flattened leaves."""
    tracker = wrap_with_attributes_tracking(Outer())
    value = tracker.inner.a
    tracker.inner = Inner(a=value + tracker.x, b=5.0)
    assert tracker.get_input_model().model_fields.keys() == {"inner.a", "x"}
    assert tracker.get_output_model().model_fields.keys() == {"inner.a", "inner.b"}


class Deep(BaseModel):
    c: float = 3.0


class Mid(BaseModel):
    a: float = 1.0
    deep: Deep = Deep()


class Top(BaseModel):
    mid: Mid = Mid()


def test_whole_nested_submodel_assignment_flattened() -> None:
    """Assigning a sub-model holding a sub-model must expand every nested leaf."""
    tracker = wrap_with_attributes_tracking(Top())
    tracker.mid = Mid(a=2.0, deep=Deep(c=4.0))
    assert not tracker.get_input_model().model_fields
    assert tracker.get_output_model().model_fields.keys() == {"mid.a", "mid.deep.c"}


class ContainerModel(BaseModel):
    lst: list[float] = [1.0, 2.0]
    d: dict[str, float] = {"k": 1.0}
    out: float = 0.0


def test_container_mutators_tracked() -> None:
    """Mutating methods of containers must be recorded as writes."""
    tracker = wrap_with_attributes_tracking(ContainerModel())
    tracker.lst.append(3.0)
    tracker.d.update(z=7.0)
    assert tracker.get_output_model().model_fields.keys() == {"lst", "d"}
    assert not tracker.get_input_model().model_fields
    assert list(tracker.lst) == [1.0, 2.0, 3.0]
    assert dict(tracker.d) == {"k": 1.0, "z": 7.0}


def test_container_aggregate_reads_tracked() -> None:
    """Reads through iteration or methods must be recorded as reads."""
    tracker = wrap_with_attributes_tracking(ContainerModel())
    total = sum(tracker.lst) + tracker.d.get("k")
    tracker.lst[0] = total
    assert tracker.get_input_model().model_fields.keys() == {"lst", "d"}
    assert tracker.get_output_model().model_fields.keys() == {"lst"}


def test_container_reads_consistent_after_write() -> None:
    """Aggregate reads must see the values written through the tracker."""
    tracker = wrap_with_attributes_tracking(ContainerModel())
    tracker.lst[0] = 100.0
    assert sum(tracker.lst) == 102.0
    assert list(tracker.lst) == [100.0, 2.0]
    assert len(tracker.lst) == 2
    assert tracker.lst == [100.0, 2.0]
    assert 100.0 in tracker.lst
    assert tracker.lst.index(2.0) == 1
    tracker.lst += [3.0]
    assert list(tracker.lst) == [100.0, 2.0, 3.0]
    del tracker.lst[0]
    assert list(tracker.lst) == [2.0, 3.0]
    assert tracker.lst.pop() == 3.0


def test_container_augmented_assignments_tracked() -> None:
    """In-place `*=` and `|=` on containers must be recorded as read and write."""
    tracker = wrap_with_attributes_tracking(ContainerModel())
    tracker.lst *= 2
    tracker.d |= {"z": 9.0}
    assert tracker.get_input_model().model_fields.keys() == {"lst", "d"}
    assert tracker.get_output_model().model_fields.keys() == {"lst", "d"}
    assert list(tracker.lst) == [1.0, 2.0, 1.0, 2.0]
    assert dict(tracker.d) == {"k": 1.0, "z": 9.0}


def test_container_inequality_and_hash() -> None:
    """A tracked container must support `!=` and hashing of immutable contents."""
    tracker = wrap_with_attributes_tracking(ContainerModel())
    assert tracker.lst != [9.0]
    immutable = wrap_with_attributes_tracking(ImmutableSequencesModel())
    assert hash(immutable.t) == hash((1, 2, 3))


class FrozenModel(BaseModel, frozen=True):
    a: float = 1.0


def test_frozen_model_write_raises() -> None:
    """Writing to a frozen model must raise like on the original model."""
    tracker = wrap_with_attributes_tracking(FrozenModel())
    with pytest.raises(ValidationError, match="frozen"):
        tracker.a = 2.0
    assert "a" not in tracker.attrs_written


class ValidatedModel(BaseModel, validate_assignment=True):
    a: int = 1


def test_assignment_validation_applies() -> None:
    """Assignment validation must run and rejected writes must not be recorded."""
    tracker = wrap_with_attributes_tracking(ValidatedModel())
    with pytest.raises(ValidationError):
        tracker.a = "not an int"
    assert "a" not in tracker.attrs_written


def test_create_tracker_class_cached() -> None:
    """The same mixin and wrapped class must yield the same tracker class."""
    first = _create_tracker_class(_ModelTrackerMixin, Main)
    second = _create_tracker_class(_ModelTrackerMixin, Main)
    assert first is second


def test_create_tracker_class_distinct_per_wrapped_class() -> None:
    """Different wrapped classes must yield different tracker classes."""
    assert _create_tracker_class(_ModelTrackerMixin, Main) is not _create_tracker_class(
        _ModelTrackerMixin, Sub
    )


def test_create_tracker_class_distinct_per_mixin() -> None:
    """Different mixins for the same wrapped class must yield different classes."""
    assert _create_tracker_class(_SequenceTracker, list) is not _create_tracker_class(
        _NDArrayTracker, list
    )


def test_tracked_models_share_class() -> None:
    """Two tracked models of the same class must share the dynamic tracker class."""
    first = wrap_with_attributes_tracking(Main(r_int=0, rw_int=0, w_int=0))
    second = wrap_with_attributes_tracking(Main(r_int=0, rw_int=0, w_int=0))
    assert type(first) is type(second)
    assert type(first).__name__ == "_TrackedMain"


def test_container_trackers_share_class(tracker) -> None:
    """Container values of the same type must reuse the same tracker class."""
    first_type = type(tracker.sub.r_sequence)
    second_type = type(tracker.sub.rw_sequence)
    assert first_type is second_type
    assert issubclass(first_type, _SequenceTracker)
