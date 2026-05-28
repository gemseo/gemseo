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
"""The discipline inputs and outputs."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

from gemseo.core.discipline.discipline_data import DisciplineData
from gemseo.core.grammars.factory import GrammarFactory
from gemseo.core.grammars.factory import GrammarType
from gemseo.core.namespaces import namespaces_separator

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.core.discipline.data_processor import DataProcessor
    from gemseo.core.grammars.base import BaseGrammar
    from gemseo.typing import MutableStrKeyMapping
    from gemseo.typing import StrKeyMapping

_GRAMMAR_FACTORY = GrammarFactory()

_DATA_DEPRECATION_MSG = (
    "`IO.data` is deprecated; use `IO.input_data` / `IO.output_data` instead."
)


class IO:
    """The discipline input and output data.

    This class defines the input and output grammar and stores the input and
    output data of the last execution.
    """

    input_data: DisciplineData
    """The input data of the last execution.

    The underlying store is mutable; changes to it are reflected immediately on
    the [IO][gemseo.core.discipline.io.IO].
    """

    output_data: DisciplineData
    """The output data of the last execution.

    The underlying store is mutable; changes to it are reflected immediately on
    the [IO][gemseo.core.discipline.io.IO].
    """

    __linear_relationships: tuple[()] | tuple[set[str], set[str]]
    """The linear relationships between the inputs and outputs.

    Expressed as `(input_names, output_names)`.
    """

    data_processor: DataProcessor | None
    """A pre- and post-processor for the discipline data.

    This mechanism requires that
    the `_run()` method of the discipline
    for which the [IO][gemseo.core.discipline.io.IO] is an attribute
    uses input data,
    returns input data
    and does not use the `input_data`, `output_data`
    or the deprecated `data` / `local_data` attributes.
    """

    input_grammar: BaseGrammar
    """The input grammar."""

    output_grammar: BaseGrammar
    """The output grammar."""

    residual_to_state_variable: dict[str, str]
    """The output variables mapping to their inputs, to be considered as residuals; they
    shall be equal to zero."""

    state_equations_are_solved: bool
    """Whether the discipline solves the state equations."""

    def __init__(
        self,
        discipline_class: type[Any],
        discipline_name: str,
        grammar_type: GrammarType,
        auto_detect_grammar_files: bool = False,
        grammar_directory: Path | str = "",
        input_grammar_file: str | Path = "",
        output_grammar_file: str | Path = "",
    ):
        """
        Args:
            discipline_class: The class of the parent discipline.
            discipline_name: The name of the parent discipline.
            grammar_type: The type of the grammars.
            auto_detect_grammar_files: Whether to find automatically the grammar.
            grammar_directory: The path to the directory where the grammar files are.
            input_grammar_file: The path to the file of the input grammar.
            output_grammar_file: The path to the file of the output grammar.
        """  # noqa: D205, D212
        self.input_grammar = _GRAMMAR_FACTORY.create(
            grammar_type,
            name=f"{discipline_name}_discipline_input",
            file_path=input_grammar_file,
            search_file=auto_detect_grammar_files,
            discipline_class=discipline_class,
            directory_path=grammar_directory,
            file_name_suffix="input",
        )
        self.output_grammar = _GRAMMAR_FACTORY.create(
            grammar_type,
            name=f"{discipline_name}_discipline_output",
            file_path=output_grammar_file,
            search_file=auto_detect_grammar_files,
            discipline_class=discipline_class,
            directory_path=grammar_directory,
            file_name_suffix="output",
        )

        self.input_data = DisciplineData()
        self.output_data = DisciplineData()
        self.data_processor = None
        self.residual_to_state_variable = {}
        self.state_equations_are_solved = False
        self.__linear_relationships = ()

    @property
    def grammar_type(self) -> GrammarType:
        """The type of grammar used for inputs and outputs."""
        return GrammarType(type(self.input_grammar).__name__)

    def prepare_input_data(self, data: StrKeyMapping) -> dict[str, Any]:
        """Prepare the input data.

        The missing input items that have default values are added,
        The items that do not exist in the input grammar are removed.

        Args:
            data: The data to be used for preparing the input data.

        Returns:
            The input data.
        """
        # TODO: remove comment: removing this line breaks one parallel test
        if not data:
            # Copy via casting in order to consistently have a dict object
            # instead of a GrammarProperty which may have side effects
            # and does not play well with the strictness of pydantic grammars.
            return dict(self.input_grammar.defaults)

        input_data = {}
        defaults = self.input_grammar.defaults

        for input_name in self.input_grammar:
            input_value = data.get(input_name)
            if input_value is not None:
                input_data[input_name] = input_value
            else:
                input_value = defaults.get(input_name)
                if input_value is not None:
                    input_data[input_name] = input_value

        return input_data

    @overload
    def get_merged_data(self, as_dict: Literal[True] = ...) -> dict[str, Any]: ...

    @overload
    def get_merged_data(self, as_dict: Literal[False]) -> DisciplineData: ...

    def get_merged_data(self, as_dict: bool = True) -> dict[str, Any] | DisciplineData:
        """Return a fresh union of the input and output stores.

        Output values override input values on overlap.

        Args:
            as_dict: Whether to return a plain `dict`.
                If `False`,
                return a
                [DisciplineData][gemseo.core.discipline.discipline_data.DisciplineData].

        Returns:
            The merged data.
        """
        merged = {**self.input_data, **self.output_data}
        if as_dict:
            return merged
        return DisciplineData(merged)

    @property
    def data(self) -> DisciplineData:
        """The merged input and output data (deprecated).

        Use [input_data][gemseo.core.discipline.io.IO.input_data] and
        [output_data][gemseo.core.discipline.io.IO.output_data] instead.

        On read, returns a fresh union of the two stores (output overrides
        input on overlap).
        On write, splits the assigned mapping into the two stores by grammar
        membership; names in neither grammar are stored on the output side.
        """
        warnings.warn(_DATA_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self.get_merged_data(as_dict=False)

    @data.setter
    def data(self, data: MutableStrKeyMapping) -> None:
        warnings.warn(_DATA_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        self._set_data_no_warn(data)

    def _set_data_no_warn(self, data: MutableStrKeyMapping) -> None:
        """Replace both stores from `data`, routing by grammar membership.

        Names in the input grammar go to `input_data`; names in the output
        grammar go to `output_data`; names in neither grammar go to
        `output_data` as a permissive catch-all matching the legacy
        behavior of the merged `data` attribute. Emits no warning so it can
        be reused by other deprecated public entry points without
        double-warning.
        """
        self.input_data = DisciplineData()
        self.output_data = DisciplineData()
        self.__route_into_stores(data)

    def __route_into_stores(self, data: StrKeyMapping) -> None:
        """Route items of `data` into `input_data` and `output_data` by grammar.

        Names in the input grammar go to `input_data`; names in the output
        grammar go to `output_data`; names in neither grammar fall into
        `output_data` as a catch-all.

        Args:
            data: The data to route. The two stores are assumed to be empty or to
                be progressively populated.
        """
        in_names = self.input_grammar
        out_names = self.output_grammar
        for key, value in data.items():
            if key in in_names:
                self.input_data[key] = value
            # Catch-all: keys outside both grammars fall into the output store,
            # matching the legacy merged `data` attribute.
            if key in out_names or key not in in_names:
                self.output_data[key] = value

    def propagate_to_input(self, updates: StrKeyMapping) -> None:
        """Copy items of `updates` whose name is an input to the input store.

        Used by MDA solvers to propagate auto-coupled outputs back into the input
        store between iterations.

        Args:
            updates: The values to propagate.
        """
        in_names = self.input_grammar
        for name, value in updates.items():
            if name in in_names:
                self.input_data[name] = value

    def get(self, name: str) -> Any:
        """Return the value of `name` from the output or input data.

        The output store is searched first; if `name` is absent, the input
        store is searched. This matches the override semantics of
        [get_merged_data][gemseo.core.discipline.io.IO.get_merged_data],
        where output values take precedence over input values on overlap.

        Args:
            name: The name of the variable.

        Returns:
            The value of the variable.

        Raises:
            KeyError: If `name` is in neither store.
        """
        if name in self.output_data:
            return self.output_data[name]
        return self.input_data[name]

    @staticmethod
    def __strip_namespaces(data: dict[str, Any], grammar: BaseGrammar) -> None:
        """Strip namespace prefixes from `data` keys in place, if any.

        Args:
            data: The dict whose keys are mutated in place.
            grammar: The grammar that owns the namespace mapping.
        """
        if grammar.to_namespaced:
            for key in tuple(data.keys()):
                if namespaces_separator in key:
                    data[key.rsplit(namespaces_separator, 1)[-1]] = data.pop(key)

    def get_input_data(self, with_namespaces: bool = True) -> dict[str, Any]:
        """Return the items of the data that are inputs.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                input names, if any.

        Returns:
            The input data.
        """
        data = self.input_data.copy()
        if not with_namespaces:
            self.__strip_namespaces(data, self.input_grammar)
        return data

    def get_output_data(self, with_namespaces: bool = True) -> dict[str, Any]:
        """Return the items of the data that are outputs.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                output names, if any.

        Returns:
            The output data.
        """
        data = self.output_data.copy()
        if not with_namespaces:
            self.__strip_namespaces(data, self.output_grammar)
        return data

    def update_output_data(self, output_data: StrKeyMapping) -> None:
        """Update the output data, taking care of the namespaces if any.

        The namespaces of the output data, if any, are automatically handled:
        if the key of an item of `output_data` is in a namespace and the key is a name
        without the namespace prefix then the item will be stored with the namespace
        prefix.

        If an item of `output_data` is not an output then it is ignored.

        Args:
            output_data: The output data to update
                [output_data][gemseo.core.discipline.io.IO.output_data] with.
        """
        out_ns = self.output_grammar.to_namespaced
        out_names = self.output_grammar
        data = self.output_data
        if out_ns:
            for key, value in output_data.items():
                if key in out_names:
                    data[key] = value
                elif key_with_ns := out_ns.get(key):
                    data[key_with_ns] = value
        else:
            for key, value in output_data.items():
                if key in out_names:
                    data[key] = value

    def initialize(self, input_data: StrKeyMapping, validate: bool) -> None:
        """Initialize the data from input data.

        Args:
            input_data: The input data.
            validate: Whether to validate `input_data`.
        """
        if validate:
            self.input_grammar.validate(input_data)

        self.input_data = DisciplineData(input_data)
        self.output_data = DisciplineData()

    def finalize(self, validate: bool) -> None:
        """Validate the output data.

        Args:
            validate: Whether to validate the (eventually post-processed) cleaned data.
        """
        if validate:
            self.output_grammar.validate(self.output_data)

    def set_linear_relationships(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Set the linear relationships between the inputs and outputs.

        Args:
            input_names: The input names in a linear relation with the outputs.
                If empty, all input names are considered.
            output_names: The output names in a linear relation with the inputs.
                If empty, all output names are considered.

        Raises:
            ValueError: If a name is not in the grammar.
        """
        input_grammar_names = self.input_grammar
        if input_names:
            input_names = set(input_names)
            self.__check_linear_relationships("input", input_names, input_grammar_names)
        else:
            input_names = set(input_grammar_names)

        output_grammar_names = self.output_grammar
        if output_names:
            output_names = set(output_names)
            self.__check_linear_relationships(
                "output", output_names, output_grammar_names
            )
        else:
            output_names = set(output_grammar_names)

        self.__linear_relationships = (input_names, output_names)

    @staticmethod
    def __check_linear_relationships(
        prefix: str,
        names: set[str],
        grammar_names: Iterable[str],
    ) -> None:
        """Check names against grammar names.

        Args:
            prefix: The kind of grammar for the error message.
            names: The names to be checked.
            grammar_names: The names in a grammar.

        Raises:
            ValueError: If a name is not in the grammar.
        """
        if alien_names := names.difference(grammar_names):
            msg = (
                f"The following {prefix}_names are not in the {prefix} grammar: "
                f"{','.join(alien_names)}."
            )
            raise ValueError(msg)

    # TODO: What is the use when the arguments are empty?
    #       How about using the same convention as for set_linear_relationships?
    def have_linear_relationships(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str],
    ) -> bool:
        """Check if an input-output restriction is linear.

        Args:
            input_names: The names of the inputs.
            output_names: The names of the outputs.

        Returns:
            Whether these outputs are linear
            with respect to these inputs.
        """
        if not self.__linear_relationships:
            return False
        return self.__linear_relationships[0].issuperset(
            input_names
        ) and self.__linear_relationships[1].issuperset(output_names)

    def __setstate__(self, state: dict[str, Any]) -> None:
        # One-way upgrade path: a legacy pickle had a single `_data` store;
        # split it by grammar membership on load. Auto-coupled keys land in
        # both stores; keys outside both grammars fall into `output_data`.
        if "_data" in state and "input_data" not in state:
            legacy = state.pop("_data")
            self.__dict__.update(state)
            self.input_data = DisciplineData()
            self.output_data = DisciplineData()
            self.__route_into_stores(legacy)
        else:
            self.__dict__.update(state)
