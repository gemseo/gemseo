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
"""Track read and written attributes of a Pydantic model at runtime.

## Design rationale
The primary use case is grammar auto-inference for
[BaseModelDiscipline][gemseo.disciplines.base_model_discipline.BaseModelDiscipline].
A discipline's inputs and outputs are determined by observing which model
fields are **read** and which are **written** during a dry-run of the user's
`_run_from_model` implementation.

## Tracking mechanism
[wrap_with_attributes_tracking][gemseo.utils.attributes_tracker.wrap_with_attributes_tracking]
is the public entry point;
it deep-copies the passed model and converts the copy in place to a
dynamically created subclass of the model class mixed with
`_ModelTrackerMixin`.
The mixin overrides `__getattribute__` / `__setattr__` to record field
accesses in a `_TrackingState` stored in the instance dictionary,
while delegating the actual attribute handling to the model class.
The tracked object therefore behaves like the original model:
`isinstance` checks pass, methods and computed fields are bound to the
tracked instance (so the fields they access are recorded), and writes go
through the regular Pydantic assignment machinery (frozen models,
`validate_assignment` and private attributes behave as usual).

Field values that are containers (nested models, sequences, NumPy arrays)
are recursively wrapped in their own trackers on first read, so that item
reads and writes are attributed to the corresponding model field.

`get_input_model` and `get_output_model` convert the collected tracking
data into Pydantic model classes that `PydanticGrammar` can consume.
"""  # noqa: E501

from __future__ import annotations

import operator
from dataclasses import dataclass
from dataclasses import field
from functools import cache
from types import new_class
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from numpy import asarray
from numpy import ndarray
from pydantic import BaseModel
from pydantic import create_model

from gemseo.utils.data_conversion import flatten_nested_dict
from gemseo.utils.pydantic import copy_field

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo
    from typing_extensions import Self

FLATTEN_SEPARATOR: Final[str] = "."

_STATE_KEY: Final[str] = "__tracking_state__"
"""The instance dictionary key holding the tracking state of a tracked model.

The instance dictionary of a Pydantic model holds the field values:
this key shall not collide with a field name and is ignored by the
Pydantic machinery (validation, serialization, copy).
"""

_MISSING: Final[object] = object()
"""A sentinel for missing instance dictionary entries."""


@dataclass(slots=True)
class _TrackingState:
    """The attribute accesses recorded for one tracked model instance."""

    attrs_read: set[str] = field(default_factory=set)
    """The read attribute names."""

    attrs_written: set[str] = field(default_factory=set)
    """The written attribute names."""

    replaced_models: dict[str, list[BaseModel]] = field(default_factory=dict)
    """The tracked sub-models replaced by whole-field assignments, by field name.

    They are kept so that the accesses recorded on them before the
    replacement still contribute to the inferred fields.
    """


class _ModelTrackerMixin:
    """Record field reads and writes of a Pydantic model.

    This mixin is combined with the class of the model to be tracked via
    `types.new_class` and the resulting class is assigned in place to a
    deep copy of the model (see `track_model`).
    """

    __slots__ = ()

    def __getattribute__(self, name: str) -> Any:
        """Record a read access and return a (possibly wrapped) field value.

        Container values are wrapped in trackers on first read and the
        wrapped value is stored back into the instance dictionary so that
        later item accesses accumulate on the same tracker.
        Non-field names (methods, computed fields, private attributes, ...)
        are handled by the model class; since they are bound to the tracked
        instance, the fields they access are recorded as well.
        """
        if name in type(self).model_fields:
            instance_dict = object.__getattribute__(self, "__dict__")
            state = instance_dict.get(_STATE_KEY)
            value = instance_dict.get(name, _MISSING)
            if state is not None and value is not _MISSING:
                state.attrs_read.add(name)
                wrapped_value = _wrap_value(value, name, state)
                if wrapped_value is not value:
                    instance_dict[name] = wrapped_value
                return wrapped_value
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Record a write access and propagate the value to the model.

        The value is set through the regular Pydantic machinery first,
        so that frozen models, assignment validation and private attributes
        behave as for the original model; the write is only recorded when
        the assignment succeeded.
        """
        instance_dict = object.__getattribute__(self, "__dict__")
        old_value = instance_dict.get(name)
        super().__setattr__(name, value)
        if name in type(self).model_fields:
            state = instance_dict.get(_STATE_KEY)
            if state is not None:
                state.attrs_written.add(name)
                if (
                    isinstance(old_value, BaseModel)
                    and _STATE_KEY in old_value.__dict__
                ):
                    # Keep the accesses recorded on the replaced sub-model.
                    state.replaced_models.setdefault(name, []).append(old_value)

    @property
    def attrs_read(self) -> set[str]:
        """The read field names."""
        return self.__dict__[_STATE_KEY].attrs_read

    @property
    def attrs_written(self) -> set[str]:
        """The written field names."""
        return self.__dict__[_STATE_KEY].attrs_written

    def get_input_model(self) -> type[BaseModel]:
        """Return a model whose fields were read during the tracked execution.

        Returns:
            A dynamically created Pydantic model class with one field per
            read attribute (→ discipline inputs), preserving the original
            `FieldInfo` metadata (type, default, validators, ...).
        """
        return self.__create_model(False)

    def get_output_model(self) -> type[BaseModel]:
        """Return a model whose fields were written during the tracked execution.

        Returns:
            A dynamically created Pydantic model class with one field per
            written attribute (→ discipline outputs), preserving the original
            `FieldInfo` metadata (type, default, validators, ...).
        """
        return self.__create_model(True)

    def __create_model(self, is_written: bool) -> type[BaseModel]:
        """Create a Pydantic model class built from the tracked field accesses.

        Args:
            is_written: Whether to collect the written fields instead of the
                read ones.

        Returns:
            The created model class.
        """
        # The keys are sorted so that the grammar field order is deterministic.
        field_definitions: dict[str, Any] = {
            n: (i.annotation, i)
            for n, i in sorted(
                flatten_nested_dict(
                    _get_tracker_data(self, is_written),
                    separator=FLATTEN_SEPARATOR,
                ).items()
            )
        }
        model = create_model("Model", **field_definitions)
        # Mark the model as created at runtime so that PydanticGrammar pickles
        # the fields information instead of the class, which cannot be pickled
        # since it has no source module to be retrieved from.
        model.__internal__ = None  # type: ignore[attr-defined]
        # TODO: This is no longer needed since pydantic 2.10, remove at some point.
        # This is another workaround for pickling a created model.
        model.__pydantic_parent_namespace__ = {}
        return model


class _BaseTracker:
    """Base class for the container trackers.

    All tracker state (`attrs_read`, `attrs_written`, `wrapped_obj`) is
    stored directly in `self.__dict__` rather than through normal attribute
    assignment, so that the initialisation does not interact with the
    attribute handling of the wrapped container classes.
    """

    attrs_read: set[str]
    """The read attribute names."""

    attrs_written: set[str]
    """The written attribute names."""

    wrapped_obj: Any
    """The wrapped object."""

    def __init__(self, obj: Any):
        """
        Args:
            obj: The object to be tracked.
        """  # noqa: D205, D212
        # Avoid infinite recursion via __getattr__.
        self.__dict__["attrs_read"] = set()
        self.__dict__["attrs_written"] = set()
        self.__dict__["wrapped_obj"] = obj


class _SequenceTracker(_BaseTracker):
    """Track read and written element accesses of a sequence (list, tuple, ...).

    When an item is accessed via `__getitem__` or `__setitem__`, the *parent*
    tracking state is notified in addition to the sequence tracker itself.
    This is necessary because `model.seq[i]` must register `seq` as
    read/written on the model — otherwise the field would not appear in the
    inferred grammar.

    The same goes for aggregate reads (iteration, `len`, `in`, equality, and
    the reading methods listed in `_READER_NAMES`) and for the mutating
    methods listed in `_WRITER_NAMES`, which all operate on the wrapped
    object.  After a mutation, the contents copied into the tracker itself
    at creation time are re-synchronized so that the operations that are not
    intercepted (e.g. `repr`) remain consistent.
    """

    SEQUENCE_MARKER: Final[str] = ""

    _READER_NAMES: Final[tuple[str, ...]] = (
        "index",
        "count",
        "get",
        "keys",
        "values",
        "items",
        "copy",
    )
    """The names of the reading methods notifying the trackers."""

    _WRITER_NAMES: Final[tuple[str, ...]] = (
        "append",
        "extend",
        "insert",
        "remove",
        "sort",
        "reverse",
        "update",
        "fill",
    )
    """The names of the mutating methods notifying the trackers."""

    _READING_WRITER_NAMES: Final[tuple[str, ...]] = (
        "pop",
        "popitem",
        "setdefault",
    )
    """The names of the methods notifying the trackers of a read and a write."""

    parent_attr_name: str
    """The name of the attribute of the parent tracking state."""

    parent_tracker: _TrackingState
    """The parent tracking state."""

    def __new__(
        cls,
        obj: Any,
        *args: Any,
    ):
        new_obj = super().__new__(cls)
        try:
            # Copy the original object state to the new one.
            obj.__class__.__init__(new_obj, obj)
        except TypeError:
            # Immutable sequences (tuple, bytes, ...) have no __init__:
            # their contents shall be passed to __new__.
            new_obj = super().__new__(cls, obj)
        return new_obj

    def __init__(
        self,
        obj: Any,
        parent_tracker: _TrackingState,
        parent_attr_name: str,
    ):
        """
        Args:
            obj: The object to be tracked.
            parent_tracker: The parent tracking state.
            parent_attr_name: The name of the parent attribute for sequence tracking.
        """  # noqa: D205, D212
        super().__init__(obj)
        self.parent_attr_name = parent_attr_name
        self.parent_tracker = parent_tracker

    def _record_read(self) -> None:
        """Notify the trackers of a read access."""
        self.parent_tracker.attrs_read.add(self.parent_attr_name)
        self.attrs_read.add(self.SEQUENCE_MARKER)

    def _record_write(self) -> None:
        """Notify the trackers of a write access."""
        self.parent_tracker.attrs_written.add(self.parent_attr_name)
        self.attrs_written.add(self.SEQUENCE_MARKER)

    def _resync_contents(self) -> None:
        """Re-synchronize the contents copied at creation from the wrapped object."""
        if isinstance(self, list):
            list.clear(self)
            list.extend(self, self.wrapped_obj)
        elif isinstance(self, dict):
            dict.clear(self)
            dict.update(self, self.wrapped_obj)

    def __getitem__(self, item: Any) -> Any:
        self._record_read()
        return self.wrapped_obj[item]

    def __setitem__(self, item: Any, value: Any) -> None:
        self._record_write()
        self.wrapped_obj[item] = value
        self._resync_contents()

    def __delitem__(self, item: Any) -> None:
        self._record_write()
        del self.wrapped_obj[item]
        self._resync_contents()

    def __iter__(self) -> Any:
        self._record_read()
        return iter(self.wrapped_obj)

    def __len__(self) -> int:
        self._record_read()
        return len(self.wrapped_obj)

    def __contains__(self, item: Any) -> bool:
        self._record_read()
        return item in self.wrapped_obj

    def __eq__(self, other: object) -> Any:
        self._record_read()
        return self.wrapped_obj == other

    def __ne__(self, other: object) -> Any:
        self._record_read()
        return self.wrapped_obj != other

    def __hash__(self) -> int:
        return hash(self.wrapped_obj)

    def __iadd__(self, other: Any) -> Self:
        # Augmented assignment also reads the current contents.
        self._record_read()
        self._record_write()
        self.__dict__["wrapped_obj"] = operator.iadd(self.wrapped_obj, other)
        self._resync_contents()
        return self

    def __imul__(self, other: Any) -> Self:
        # Augmented assignment also reads the current contents.
        self._record_read()
        self._record_write()
        self.__dict__["wrapped_obj"] = operator.imul(self.wrapped_obj, other)
        self._resync_contents()
        return self

    def __ior__(self, other: Any) -> Self:
        # Augmented assignment also reads the current contents.
        self._record_read()
        self._record_write()
        self.__dict__["wrapped_obj"] = operator.ior(self.wrapped_obj, other)
        self._resync_contents()
        return self


def _make_tracking_method(name: str, read: bool, write: bool) -> Any:
    """Create a method delegating to the wrapped object with access recording.

    Args:
        name: The name of the method of the wrapped object.
        read: Whether the method reads the contents.
        write: Whether the method mutates the contents.

    Returns:
        The created method.
    """

    def method(self: _SequenceTracker, *args: Any, **kwargs: Any) -> Any:
        if read:
            self._record_read()
        if write:
            self._record_write()
        result = getattr(self.wrapped_obj, name)(*args, **kwargs)
        if write:
            self._resync_contents()
        return result

    method.__name__ = name
    return method


for _name in _SequenceTracker._READER_NAMES:
    setattr(_SequenceTracker, _name, _make_tracking_method(_name, True, False))
for _name in _SequenceTracker._WRITER_NAMES:
    setattr(_SequenceTracker, _name, _make_tracking_method(_name, False, True))
for _name in _SequenceTracker._READING_WRITER_NAMES:
    setattr(_SequenceTracker, _name, _make_tracking_method(_name, True, True))


class _NDArrayTracker(_SequenceTracker):
    """Track read and written accesses of a NumPy array.

    The tracker is a view sharing the memory of the wrapped array
    (see https://numpy.org/doc/stable/user/basics.subclassing.html):

    * views derived from it (rows, slices, transposes) inherit the tracking
      via `__array_finalize__`, so writes through them are recorded;
    * ufunc calls record reads of the tracked operands and a write when a
      tracked array is the `out=` target (in-place operations included);
      they are computed on the base arrays so that their results are plain
      arrays.
    """

    def __new__(
        cls,
        input_array: ndarray,
        *args: Any,
    ):
        return asarray(input_array).view(cls)

    def __array_finalize__(self, obj: Any) -> None:
        if (
            isinstance(obj, _NDArrayTracker)
            and "parent_tracker" in obj.__dict__
            and "parent_tracker" not in self.__dict__
        ):
            # A view created from a tracked array (row, slice, transpose, ...)
            # shares its memory: attribute the accesses to the same field by
            # sharing the records of the source tracker.
            self.__dict__["attrs_read"] = obj.attrs_read
            self.__dict__["attrs_written"] = obj.attrs_written
            self.__dict__["wrapped_obj"] = self.view(ndarray)
            self.__dict__["parent_tracker"] = obj.parent_tracker
            self.__dict__["parent_attr_name"] = obj.parent_attr_name

    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> Any:
        out = kwargs.get("out", ())
        for item in inputs:
            if _is_tracking(item):
                item._record_read()
        for item in out:
            if _is_tracking(item):
                item._record_write()
        if out:
            kwargs["out"] = tuple(
                item.view(ndarray) if isinstance(item, _NDArrayTracker) else item
                for item in out
            )
        inputs = tuple(
            item.view(ndarray) if isinstance(item, _NDArrayTracker) else item
            for item in inputs
        )
        # Computing on the base arrays makes the results plain arrays.
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getitem__(self, item: Any) -> Any:
        self._record_read()
        # Indexing the tracker itself (instead of the wrapped array) returns
        # tracked views for rows and slices, so that writes through them are
        # recorded.
        return ndarray.__getitem__(self, item)

    def __setitem__(self, item: Any, value: Any) -> None:
        self._record_write()
        # Writing through the base class view (which shares the memory of the
        # tracker) avoids the internal item reads that NumPy performs on the
        # subclass for some assignments (e.g. with an ellipsis), which would
        # be recorded as read accesses.
        self.wrapped_obj[item] = value

    def __iter__(self) -> Any:
        self._record_read()
        # Iterating the tracker itself yields tracked row views.
        return ndarray.__iter__(self)


def _is_tracking(obj: Any) -> bool:
    """Return whether an object is a tracker bound to a tracking state.

    Args:
        obj: The object to check.
    """
    return isinstance(obj, _BaseTracker) and "parent_tracker" in obj.__dict__


def track_model(model: BaseModel) -> BaseModel:
    """Convert a model in place into a tracked subclass instance.

    Unlike
    [wrap_with_attributes_tracking][gemseo.utils.attributes_tracker.wrap_with_attributes_tracking],
    the model is not copied: the passed instance itself becomes tracked.

    Args:
        model: The model instance to be tracked; it is modified in place.

    Returns:
        The same instance, now an instance of a dynamically created
        subclass of its class mixed with `_ModelTrackerMixin`.
    """
    _check_model_type(model)
    model.__class__ = _create_tracker_class(_ModelTrackerMixin, type(model))  # type: ignore[assignment]
    model.__dict__[_STATE_KEY] = _TrackingState()
    return model


@cache
def _create_tracker_class(mixin_class: type, wrapped_class: type) -> type:
    """Return the class combining a tracking mixin with a wrapped class.

    The classes are cached: creating a class is costly and the same
    combination is requested for every tracked instance of a class.

    Args:
        mixin_class: The tracking mixin class.
        wrapped_class: The class of the object to be tracked.

    Returns:
        The combined class.
    """
    name = (
        f"_Tracked{wrapped_class.__name__}"
        if issubclass(mixin_class, _ModelTrackerMixin)
        else "tracker_class"
    )
    return new_class(name, (mixin_class, wrapped_class))


def _wrap_value(
    value: Any,
    attr_name: str,
    state: _TrackingState,
) -> Any:
    """Wrap a field value so that its own accesses are tracked.

    Args:
        value: The field value to be wrapped.
        attr_name: The name of the field holding the value.
        state: The tracking state of the model holding the field.

    Returns:
        The wrapped value, or the value itself when it needs no wrapping.
    """
    if isinstance(value, _BaseTracker):
        # Already wrapped on a previous read.
        return value
    if isinstance(value, BaseModel):
        if _STATE_KEY not in value.__dict__:
            track_model(value)
        return value
    if isinstance(value, ndarray):
        mixin_class = _NDArrayTracker
    elif hasattr(value, "__getitem__") and not isinstance(value, str):
        mixin_class = _SequenceTracker
    else:
        return value
    return _create_tracker_class(mixin_class, type(value))(value, state, attr_name)


def _get_field_info(model: BaseModel, attr_name: str) -> FieldInfo:
    """Return the FieldInfo of a field of a tracked model.

    Args:
        model: The tracked model.
        attr_name: The field name.
    """
    value = _unwrap_value(model.__dict__[attr_name])
    # The default factory, if any, is discarded since a default is set:
    # a field cannot have both.
    return copy_field(attr_name, type(model), default=value, default_factory=None)


def _unwrap_value(value: Any) -> Any:
    """Return the original object wrapped in a container tracker.

    Args:
        value: A possibly wrapped field value.
    """
    if isinstance(value, _BaseTracker):
        return value.wrapped_obj
    return value


def _get_tracker_data(model: BaseModel, is_written: bool) -> dict[str, Any]:
    """Return field data collected for a tracked model, filtered by direction.

    The function handles three cases:

    a. **Leaf written fields** (`is_written=True`): attributes that appear in
       the written records are added directly with their `FieldInfo`.
    b. **Leaf read fields** (`is_written=False`): attributes that appear in
       the read records but whose value is *not* another tracker are added
       with their `FieldInfo`.
    c. **Nested model fields**: if an attribute's value is a tracked model
       (i.e. the field holds a sub-model), the function recurses into it.
       If the recursion yields no fields, the field is excluded — this
       avoids exposing a sub-model as an input when it was only written to,
       or when only its methods were called.

    Args:
        model: The tracked model to inspect.
        is_written: If `True`, collect written (output) fields;
            if `False`, collect read (input) fields.

    Returns:
        A possibly nested mapping of `{field_name: FieldInfo | dict}`,
        where nested dicts correspond to sub-models.
    """
    state = model.__dict__[_STATE_KEY]
    items = {}

    if is_written:
        for attr_name in state.attrs_written:
            attr_value = _unwrap_value(model.__dict__[attr_name])
            if isinstance(attr_value, BaseModel):
                # A whole sub-model assignment writes all its fields.
                items[attr_name] = _expand_model_fields(attr_value)
            else:
                items[attr_name] = _get_field_info(model, attr_name)

    for attr_name in state.attrs_read:
        attr_value = model.__dict__[attr_name]
        is_submodel = isinstance(attr_value, BaseModel)
        tracker_data = {}
        if isinstance(attr_value, _SequenceTracker):
            if attr_value.attrs_written and not attr_value.attrs_read:
                # The attribute was only written to.
                continue
        elif is_submodel and _STATE_KEY in attr_value.__dict__:
            tracker_data = _get_tracker_data(attr_value, is_written)
        if not is_written:
            # The accesses recorded on sub-models replaced by whole-field
            # assignments contribute to the read fields.
            for replaced_model in state.replaced_models.get(attr_name, ()):
                replaced_data = _get_tracker_data(replaced_model, is_written)
                replaced_data.update(tracker_data)
                tracker_data = replaced_data
        if not tracker_data:
            if is_written or is_submodel:
                # Written leaves are collected above, and a sub-model without
                # relevant accesses (only written to, or only its methods
                # were called) is excluded.
                continue
            tracker_data = _get_field_info(model, attr_name)
        existing_data = items.get(attr_name)
        if isinstance(existing_data, dict) and isinstance(tracker_data, dict):
            existing_data.update(tracker_data)
        else:
            items[attr_name] = tracker_data

    return items


def _expand_model_fields(model: BaseModel) -> dict[str, Any]:
    """Return the FieldInfo of every field of a model, expanding nested models.

    Args:
        model: The model.

    Returns:
        A possibly nested mapping of `{field_name: FieldInfo | dict}`.
    """
    items = {}
    for field_name in type(model).model_fields:
        value = _unwrap_value(model.__dict__[field_name])
        if isinstance(value, BaseModel):
            items[field_name] = _expand_model_fields(value)
        else:
            items[field_name] = _get_field_info(model, field_name)
    return items


def wrap_with_attributes_tracking(obj: BaseModel) -> BaseModel:
    """Wrap a Pydantic model so that attribute reads and writes are recorded.

    Pass the returned tracked copy to user code instead of the original model.
    The tracked copy behaves like the original model (`isinstance` checks,
    methods, computed fields and validation are unaffected) while attribute
    accesses are silently recorded.  Call `get_input_model()` on it to obtain
    a Pydantic model class whose fields are the **read** attributes (inputs),
    and `get_output_model()` for the **written** attributes (outputs).

    Args:
        obj: The Pydantic `BaseModel` instance to track.

    Returns:
        A tracked deep copy of *obj*.
    """
    _check_model_type(obj)
    return track_model(obj.model_copy(deep=True))


def _check_model_type(obj: Any) -> None:
    """Check that an object is a Pydantic model instance.

    Args:
        obj: The object to check.

    Raises:
        TypeError: If the object is not a Pydantic model instance.
    """
    if not isinstance(obj, BaseModel):
        msg = (
            "The object to be tracked shall be a Pydantic BaseModel instance; "
            f"got {type(obj).__name__}."
        )
        raise TypeError(msg)
