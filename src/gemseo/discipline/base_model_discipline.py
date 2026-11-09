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
"""Base class for implementing disciplines based on a Pydantic model."""

from __future__ import annotations

import logging
from abc import abstractmethod
from copy import deepcopy
from operator import attrgetter
from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.discipline import Discipline
from gemseo.core.grammar.pydantic import PydanticGrammar
from gemseo.util.attributes_tracker import copy_untracked_value
from gemseo.util.attributes_tracker import flatten_separator
from gemseo.util.attributes_tracker import wrap_with_attributes_tracking

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

    from gemseo.util.typing import StrKeyMapping

logger = logging.getLogger(__name__)


class BaseModelDiscipline(Discipline):
    """A discipline whose inputs and outputs are inferred from a Pydantic model.

    Subclass this class to implement a discipline by writing
    `_run_from_model` instead of the usual `_run`.

    At construction time,
    `_run_from_model` is called once with a tracked copy of *model* (see
    [wrap_with_attributes_tracking][gemseo.util.attributes_tracker.wrap_with_attributes_tracking]).
    The tracking layer records which model fields are **read before having
    been assigned as a whole** (→ discipline inputs) and which are
    **written** (→ discipline outputs).
    Fields that are neither read nor written are ignored.
    The outputs computed by this dry-run are then written back into *model* itself:
    a sub-model assigned as a whole, e.g. `model.y = Y(...)`,
    is written back as a whole,
    and the other outputs are written back field by field.
    The outputs are written back without validation:
    a model validating the assignments has validated them during the dry-run,
    whereas a model not validating the assignments has not validated them at all.
    The field validators of the models holding the written fields are therefore
    not rerun at write-back, so they do not transform the values written back.
    Prefer explicit statements in `_run_from_model` to field validators for the outputs:
    a warning is logged at construction when such a model defines field validators.
    Pydantic rejects the assignment of a frozen field or of a field of a frozen model,
    so such a field can only be changed in place during the dry-run,
    e.g. `model.y[0] = ...`.
    As the discipline sets its inputs on a copy of *model* at execution,
    a frozen field that is an input of the discipline
    and a frozen model holding an input of the discipline
    are rejected at construction with a `ValueError`.
    A frozen field that is not an input
    or a field of a frozen model holding no input,
    when changed in place during the dry-run,
    is not written back:
    *model* keeps its previous value,
    which the disciplines built next see
    and which is also the output default of the discipline for this field.
    This field remains an output of the discipline.
    The input defaults of the discipline itself are the values of *model*
    from before this write-back.
    This is how a model shared by several disciplines cascades values forward:
    the disciplines built first compute into the shared model
    the fields a user could only initialize with a dummy value,
    so that the disciplines built next dry-run on sensible values instead,
    which also affects which fields they infer as inputs and outputs.
    **The order in which the disciplines are built therefore matters.**
    For the same reason, a model must not be reused for a second build:
    a field both read and written by `_run_from_model`, e.g. `model.x = model.x + 1`,
    would drift at each build.

    Executing the discipline, on the other hand, never modifies *model*:
    `_run_from_model` is then called on a deep copy of it.

    A field computed and then re-used as an intermediate value is an output
    only: its value comes from the discipline, not from another one.
    A field read before being written is both an input and an output,
    e.g. `model.y = model.y + 1`.
    A field whose elements only are written, e.g. `model.y[0] = ...`, does not
    lose the reads that follow: unlike a field assigned as a whole, it stays an
    input when it is read as well.

    Nested models are supported: a field `y_1` on a sub-model `y` becomes
    the flat name `"y.y_1"` in the grammar.

    Every input and output field must be read and written as an attribute of
    *model*; binding a field to a local name defeats the tracking.
    See the how-to guide
    [Create a discipline from a Pydantic model][]
    for the tracking mechanism, this limitation and nested-model handling.

    The write-back copies each output separately and has two limitations.
    Two outputs referring to the same mutable object during the dry-run,
    e.g. `model.a = model.b = shared`,
    are written back as two independent copies,
    so that an in-place change of one no longer shows through the other.
    A sub-model read during the dry-run and held by a container
    other than a list, a tuple, a set, a frozenset or a dictionary,
    e.g. a `collections.deque`,
    is written back with its tracking layer.

    Subclassing:
    1. Implement
       `_run_from_model`:
       read inputs from *model*, compute, and write outputs back into *model*.
    2. If `__init__` stores extra state used in
       `_run_from_model`,
       set that state **before** calling `super().__init__(model)`,
       because grammar detection runs during the parent constructor.
    """

    _model: BaseModel
    """The model defining the inputs and outputs of the discipline."""

    __output_getters: dict[str, Callable[[Any], Any]]
    """The accessors reading each output value from the model, by grammar name."""

    __input_setters: dict[str, tuple[Callable[[Any], Any] | None, str]]
    """The parent accessor and leaf name setting each input, by grammar name.

    The parent accessor is `None` when the input is a top-level field.
    """

    def __init__(
        self,
        model: BaseModel,
        name: str = "",
    ) -> None:
        """
        Args:
            model: The model.

        Raises:
            ValueError: When an input of the discipline
                or the model holding it is frozen.
        """  # noqa: D205, D212
        super().__init__(name=name)
        self._model = model
        self._init_grammars()

    def _init_grammars(self) -> None:
        """Detect input and output fields by dry-running `_run_from_model`.

        A tracked deep copy of the stored model is passed to
        `_run_from_model`.
        Fields that are **read before having been assigned as a whole** during
        that call become discipline inputs;
        fields that are **written** become discipline outputs.
        The resulting grammars replace `input_grammar` and `output_grammar`.
        The input defaults are taken from the stored model,
        then the outputs of the dry-run are written into the stored model itself
        (see the class docstring)
        and the output defaults are taken from it.

        Raises:
            ValueError: When an input of the discipline
                or the model holding it is frozen.
        """
        # The tracked model is a deep copy: the dry-run itself cannot modify
        # the stored model; its outputs are written back into it below.
        tracked_model = wrap_with_attributes_tracking(self._model)
        self._run_from_model(tracked_model)
        # The grammar names follow the convention used by IO, so that
        # validation messages identify the discipline.
        self.io.input_grammar = PydanticGrammar(
            f"{self.name}_input", model=tracked_model.get_input_model()
        )
        self.io.output_grammar = PydanticGrammar(
            f"{self.name}_output", model=tracked_model.get_output_model()
        )
        # Pre-compute the dotted-path accessors used at every execution.
        self.__output_getters = {
            name: attrgetter(name) for name in self.io.output_grammar
        }
        self.__input_setters = {}
        for name in self.io.input_grammar:
            parent_path, _, leaf_name = name.rpartition(flatten_separator)
            self.__input_setters[name] = (
                attrgetter(parent_path) if parent_path else None,
                leaf_name,
            )
        # The input defaults are captured from the stored model before the
        # dry-run outputs are written into it,
        # so that a field both read and written by this discipline
        # does not get its own dry-run output as input default.
        # Deep copies decouple the defaults from the stored model.
        self.__set_defaults(self.io.input_grammar)
        # All the checks are done before any write,
        # so that a failed check leaves the stored model untouched.
        # The write-back itself is not atomic:
        # a copy raising an error, e.g. for a value holding a lock or an open file,
        # leaves the outputs written before it in the stored model.
        self.__check_inputs()
        write_back_paths = self.__get_write_back_paths(tracked_model)
        self.__warn_about_field_validators(write_back_paths)
        for path in write_back_paths:
            value = attrgetter(path)(tracked_model)
            parent, leaf_name = self.__get_parent(path)
            # The values are written without validation:
            # validating them again would run the field validators twice
            # and a model validator on intermediate states
            # mixing written-back values and values not written back yet.
            # The dry-run has validated them only when the model validates assignments.
            parent.__dict__[leaf_name] = copy_untracked_value(value)
            parent.__pydantic_fields_set__.add(leaf_name)
        # The output defaults are captured after the write-back,
        # so that they hold the values computed by the dry-run,
        # except for the frozen fields, which are not written back.
        self.__set_defaults(self.io.output_grammar)

    def __set_defaults(self, grammar: PydanticGrammar) -> None:
        """Set the defaults of a grammar from the values of the stored model.

        Args:
            grammar: The grammar whose defaults are set.
        """
        for name in grammar.defaults:
            grammar.defaults[name] = deepcopy(self._get_attr_val(name, self._model))

    def __get_write_back_paths(self, tracked_model: BaseModel) -> list[str]:
        """Return the paths of the fields to write back into the stored model.

        An output is written back through its outermost sub-model
        assigned as a whole during the dry-run, e.g. `model.y = Y(...)`,
        and through its own field otherwise.
        A field that is frozen or held by a frozen model is not written back:
        as Pydantic rejected its assignment during the dry-run,
        it can only have been changed in place, e.g. `model.y[0] = ...`.

        Args:
            tracked_model: The tracked model passed to the dry-run.

        Returns:
            The paths of the fields to write back.
        """
        # A dictionary removes the duplicates while keeping the order,
        # since several outputs can be written back through the same sub-model.
        paths = {}
        for name in self.io.output_grammar:
            tracked_parent = tracked_model
            segments = []
            for segment in name.split(flatten_separator):
                segments.append(segment)
                if segment in tracked_parent.attrs_assigned:
                    break
                tracked_parent = tracked_parent.__dict__[segment]
            path = flatten_separator.join(segments)
            parent, leaf_name = self.__get_parent(path)
            if not self.__is_frozen(parent, leaf_name):
                paths[path] = None
        return list(paths)

    def __warn_about_field_validators(self, write_back_paths: list[str]) -> None:
        """Log a warning when a model of the written fields defines field validators.

        The field validators are not rerun on the values written back.

        Args:
            write_back_paths: The paths of the fields written back.
        """
        # A dictionary removes the duplicates while keeping the order.
        model_classes = {}
        for path in write_back_paths:
            parent, _ = self.__get_parent(path)
            model_classes[type(parent)] = None

        names = [
            model_class.__name__
            for model_class in model_classes
            if model_class.__pydantic_decorators__.field_validators
        ]
        if names:
            logger.warning(
                "The field validators of the models %s are not rerun "
                "when the outputs of the discipline %s are written back "
                "into the model at construction; "
                "prefer explicit statements in _run_from_model.",
                ", ".join(names),
                self.name,
            )

    def __get_parent(self, path: str) -> tuple[BaseModel, str]:
        """Return the model of the stored model holding an output field and its name.

        Args:
            path: The path of the output field.

        Returns:
            The model holding the field and the name of the field.
        """
        parent_path, _, leaf_name = path.rpartition(flatten_separator)
        if parent_path:
            return attrgetter(parent_path)(self._model), leaf_name
        return self._model, leaf_name

    def __check_inputs(self) -> None:
        """Check that the inputs can be set on a copy of the stored model.

        All the problems are reported at once.

        Raises:
            ValueError: When an input or the model holding it is frozen.
        """
        problems = []
        for name, (parent_getter, leaf_name) in self.__input_setters.items():
            parent = (
                self._model if parent_getter is None else parent_getter(self._model)
            )
            if not self.__is_frozen(parent, leaf_name):
                continue

            parent_name = type(parent).__name__
            if type(parent).model_config.get("frozen"):
                problems.append(f"the model {parent_name} holding the input {name}")
            else:
                problems.append(
                    f"the field {leaf_name} of the model {parent_name} "
                    f"holding the input {name}"
                )

        if problems:
            msg = (
                "The discipline sets its inputs on a copy of the model at execution, "
                "so the following must not be frozen:\n"
                + "\n".join(f"- {problem}" for problem in problems)
            )
            raise ValueError(msg)

    @staticmethod
    def __is_frozen(parent: BaseModel, leaf_name: str) -> bool:
        """Check whether a field or the model holding it is frozen.

        Args:
            parent: The model holding the field.
            leaf_name: The name of the field.

        Returns:
            Whether the field or the model holding it is frozen.
        """
        parent_class = type(parent)
        return bool(
            parent_class.model_config.get("frozen")
            or parent_class.model_fields[leaf_name].frozen
        )

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping:
        """Apply input data to the model and delegate to `_run_from_model`.

        Returns:
            The output values keyed by grammar name.
        """
        model = self._get_model(input_data)
        self._run_from_model(model)
        return self._get_output_data(model)

    @abstractmethod
    def _run_from_model(self, model: BaseModel) -> None:
        """Execute the discipline computation using the given model.

        Args:
            model: The model containing the current input values;
                write output values back into it.
        """

    def _get_model(self, input_data: StrKeyMapping) -> BaseModel:
        """Return a deep copy of the stored model with input values applied.

        Args:
            input_data: The current input values keyed by grammar name
                (dot-separated paths for nested fields).

        Returns:
            A model copy ready to be passed to `_run_from_model`.
        """
        model = self._model.model_copy(deep=True)
        # The values are deep-copied because they may be the grammar default
        # values themselves, which shall not be modified when
        # _run_from_model writes into the model in-place.
        for name, value in input_data.items():
            parent_getter, leaf_name = self.__input_setters[name]
            parent = model if parent_getter is None else parent_getter(model)
            setattr(parent, leaf_name, deepcopy(value))
        return model

    def _get_output_data(self, model: BaseModel) -> StrKeyMapping:
        """Collect output values from the model after `_run_from_model` returns.

        Args:
            model: The model that was passed to `_run_from_model`.

        Returns:
            The output values keyed by grammar name.
        """
        data = {}
        for name, getter in self.__output_getters.items():
            data[name] = getter(model)
        return data

    @staticmethod
    def _get_attr_val(attr_path: str, obj: Any) -> Any:
        """Return a (possibly nested) attribute value from *obj* by dot-separated path.

        Args:
            attr_path: A dot-separated attribute path, e.g. `"y.y_1"`.
            obj: The object from which to read the attribute.

        Returns:
            The value of the leaf attribute.
        """
        return attrgetter(attr_path)(obj)
