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

from abc import abstractmethod
from copy import deepcopy
from operator import attrgetter
from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.discipline import Discipline
from gemseo.core.grammars.pydantic import PydanticGrammar
from gemseo.utils.attributes_tracker import FLATTEN_SEPARATOR
from gemseo.utils.attributes_tracker import wrap_with_attributes_tracking

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

    from gemseo.typing import StrKeyMapping


class BaseModelDiscipline(Discipline):
    """A discipline whose inputs and outputs are inferred from a Pydantic model.

    Subclass this class to implement a discipline by writing
    `_run_from_model` instead of the usual `_run`.

    At construction time,
    `_run_from_model` is called once with a tracked copy of *model* (see
    [wrap_with_attributes_tracking][gemseo.utils.attributes_tracker.wrap_with_attributes_tracking]).
    The tracking layer records which model fields are **read** (→ discipline inputs)
    and which are **written** (→ discipline outputs).
    Fields that are neither read nor written are ignored.

    Nested models are supported: a field `y_1` on a sub-model `y` becomes
    the flat name `"y.y_1"` in the grammar.

    Every input and output field must be read and written as an attribute of
    *model*; binding a field to a local name defeats the tracking.
    See the how-to guide
    [Create a discipline from a Pydantic model][]
    for the tracking mechanism, this limitation and nested-model handling.

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
        """  # noqa: D205, D212
        super().__init__(name=name)
        self._model = model
        self._init_grammars()

    def _init_grammars(self) -> None:
        """Detect input and output fields by dry-running `_run_from_model`.

        A tracked deep copy of the stored model is passed to
        `_run_from_model`.
        Fields that are **read** during that call become discipline inputs;
        fields that are **written** become discipline outputs.
        The resulting grammars replace `input_grammar` and `output_grammar`.
        """
        # The tracked model is a deep copy: the dry-run cannot modify the
        # stored model.
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
        # The grammar defaults are captured from the copy after the dry-run,
        # which may have written into it: restore them from the stored model.
        # Deep copies decouple the defaults from the stored model.
        for grammar in (self.io.input_grammar, self.io.output_grammar):
            for name in grammar.defaults:
                grammar.defaults[name] = deepcopy(self._get_attr_val(name, self._model))
        # Pre-compute the dotted-path accessors used at every execution.
        self.__output_getters = {
            name: attrgetter(name) for name in self.io.output_grammar
        }
        self.__input_setters = {}
        for name in self.io.input_grammar:
            parent_path, _, leaf_name = name.rpartition(FLATTEN_SEPARATOR)
            self.__input_setters[name] = (
                attrgetter(parent_path) if parent_path else None,
                leaf_name,
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
