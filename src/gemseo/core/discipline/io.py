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

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.discipline.discipline_data import DisciplineData
from gemseo.core.grammars.factory import GrammarFactory
from gemseo.core.grammars.factory import GrammarType
from gemseo.core.namespaces import namespaces_separator

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.core.discipline.data_processor import DataProcessor
    from gemseo.core.grammars.base_grammar import BaseGrammar
    from gemseo.typing import MutableStrKeyMapping
    from gemseo.typing import StrKeyMapping

_GRAMMAR_FACTORY = GrammarFactory()


class IO:
    """The discipline input and output data.

    This class defines the input and output grammar and stores the input and
    output data of the last execution.
    """

    __data: DisciplineData
    """The input and output data."""

    __linear_relationships: tuple[()] | tuple[set[str], set[str]]
    """The linear relationships between the inputs and outputs.

    Expressed as ``(input_names, output_names)``.
    """

    data_processor: DataProcessor | None
    """A pre- and post-processor for the discipline data.

    This mechanism requires that
    the ``_run()`` method of the discipline
    for which the :class:`.IO` is an attribute
    uses input data,
    returns input data
    and does not use the ``io.data`` or ``local_data`` attributes.
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

        self.data = {}
        self.data_processor = None
        self.residual_to_state_variable = {}
        self.state_equations_are_solved = False
        self.__linear_relationships = ()

    @property
    def grammar_type(self) -> GrammarType:
        """The type of grammar used for inputs and outputs."""
        return GrammarType(type(self.input_grammar).__name__)

    def prepare_input_data(self, data: StrKeyMapping) -> StrKeyMapping:
        """Prepare the input data.

        The missing input items that have default values are added,
        The items that do not exist in the input grammar are removed.

        Args:
            data: The data to be used for preparing the input data.

        Returns:
            The input data.
        """
        if not data:
            return self.input_grammar.defaults.copy()

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

    @property
    def data(self) -> DisciplineData:
        """The current input and output data.

        When set, the passed data are shallow copied.
        """
        return self.__data

    @data.setter
    def data(self, data: MutableStrKeyMapping) -> None:
        self.__data = DisciplineData(data)

    def __get_data(self, with_namespaces: bool, grammar: BaseGrammar) -> dict[str, Any]:
        """Return the local data restricted to the items in a grammar.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                output names, if any.
            grammar: The grammar that provides the names to be restricted to.

        Returns:
            The local output data.
        """
        copy_ = self.data.copy()
        for name in copy_.keys() - grammar.keys():
            del copy_[name]

        if not with_namespaces and grammar.to_namespaced:
            for key in tuple(copy_.keys()):
                copy_[key.rsplit(namespaces_separator, 1)[-1]] = copy_.pop(key)

        return copy_

    def get_input_data(self, with_namespaces: bool = True) -> dict[str, Any]:
        """Return the items of the data that are inputs.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                input names, if any.

        Returns:
            The input data.
        """
        return self.__get_data(with_namespaces, self.input_grammar)

    def get_output_data(self, with_namespaces: bool = True) -> dict[str, Any]:
        """Return the items of the data that are outputs.

        Args:
            with_namespaces: Whether to keep the namespace prefix of the
                output names, if any.

        Returns:
            The output data.
        """
        return self.__get_data(with_namespaces, self.output_grammar)

    def update_output_data(self, output_data: StrKeyMapping) -> None:
        """Update the output in data, taking care of the namespaces if any.

        The namespaces of the output data, if any, are automatically handled:
        if the key of an item of ``output_data`` is in a namespace and the key is a name
        without the namespace prefix then the item will be stored with the namespace
        prefix.

        If an item of ``output_data`` is not an output then it is ignored.

        Args:
            output_data: The output data to update :attr:`.data` with.
        """
        out_ns = self.output_grammar.to_namespaced
        out_names = self.output_grammar
        data = self.__data
        for key, value in output_data.items():
            if key in out_names:
                data[key] = value
            elif key_with_ns := out_ns.get(key):
                data[key_with_ns] = value

    def initialize(self, input_data: StrKeyMapping, validate: bool) -> None:
        """Initialize the data from input data.

        Args:
            input_data: The input data.
            validate: Whether to validate ``input_data``.
        """
        if validate:
            self.input_grammar.validate(input_data)

        self.data = input_data

    def finalize(self, validate: bool) -> None:
        """Validate the output data.

        Args:
            validate: Whether to validate the (eventually post-processed) cleaned data.
        """
        if validate:
            self.output_grammar.validate(self.data)

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
