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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                        documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A class to create function from a discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.function.discipline_adapter import DisciplineAdapter
from gemseo.util.constant import read_only_empty_dict

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.core.discipline import Discipline
    from gemseo.core.grammar.base import BaseGrammar
    from gemseo.space.variable import CategoricalVariable


class DisciplineAdapterGenerator:
    """A generator of discipline adapter.

    Given a discipline,
    a
    [DisciplineAdapter][gemseo.core.function.discipline_adapter.DisciplineAdapter]
    computes specific outputs from specific inputs.
    """

    discipline: Discipline
    """The discipline from which to generate discipline adapters."""

    _adapter_class: ClassVar[type[DisciplineAdapter]] = DisciplineAdapter
    """The class of discipline adapter generated."""

    __name_to_size: MutableMapping[str, int]
    """The map from an input name to its size, if known."""

    __name_to_categorical_variable: Mapping[str, CategoricalVariable]
    """The map from the name of a categorical input to its variable."""

    def __init__(
        self,
        discipline: Discipline,
        name_to_size: MutableMapping[str, int] = read_only_empty_dict,
        name_to_categorical_variable: Mapping[
            str, CategoricalVariable
        ] = read_only_empty_dict,
    ) -> None:
        """
        Args:
            discipline: The discipline from which to generate discipline adapters.
            name_to_size: The sizes of the input variables.
                If empty,
                determine them from the default inputs and local data of the discipline.
            name_to_categorical_variable: The map
                from the name of a categorical input to its variable;
                the discipline receives the label of the category
                instead of its position in the input vector.
                If empty, no input is categorical.
        """  # noqa: D205, D212, D415
        self.discipline = discipline
        self.__name_to_size = name_to_size or {}
        self.__name_to_categorical_variable = dict(name_to_categorical_variable)

    def get_function(
        self,
        input_names: Sequence[str],
        output_names: Sequence[str],
        default_input_data: Mapping[str, ndarray] = read_only_empty_dict,
        is_differentiable: bool = True,
        differentiated_input_names_substitute: Sequence[str] = (),
    ) -> DisciplineAdapter:
        """Build a function executing a discipline for some inputs and outputs.

        Args:
            input_names: The discipline input names defining the function input vector.
                If empty,
                use all the discipline inputs.
            output_names: The discipline output names
                defining the function output vector.
                If empty,
                use all the discipline outputs.
            default_input_data: The default values of the input variables.
                If empty,
                use the default input values of the discipline.
            is_differentiable: Whether the function is differentiable.
            differentiated_input_names_substitute: The names of the inputs
                with respect to which to differentiate the functions.
                If empty,
                use `input_names`.
                This argument is not used when `is_differentiable` is `False`.

        Returns:
            The function.

        Raises:
            ValueError: When either
                an input name is not a discipline input name,
                a differentiated input name is not a discipline input name
                or an output name is not a discipline output name.
        """
        input_names = self.__get_names(
            "inputs",
            input_names,
            self.discipline.io.input_grammar,
        )
        output_names = self.__get_names(
            "outputs",
            output_names,
            self.discipline.io.output_grammar,
        )
        if differentiated_input_names_substitute:
            self.__get_names(
                "inputs",
                differentiated_input_names_substitute,
                self.discipline.io.input_grammar,
            )
        else:
            # A categorical input is not differentiable,
            # its label is not a number.
            differentiated_input_names_substitute = [
                name
                for name in input_names
                if name not in self.__name_to_categorical_variable
            ]

        if is_differentiable:
            # An empty list would mean all the inputs: when all of them are
            # categorical, no input is differentiated.
            if differentiated_input_names_substitute:
                self.discipline.add_differentiated_inputs(
                    differentiated_input_names_substitute
                )

            self.discipline.add_differentiated_outputs(output_names)

        return self._adapter_class(
            input_names,
            output_names,
            default_input_data or {},
            self.discipline,
            self.__name_to_size,
            differentiated_input_names_substitute=differentiated_input_names_substitute,
            name_to_categorical_variable=self.__name_to_categorical_variable,
        )

    def __get_names(
        self,
        group_name: str,
        names: Sequence[str],
        grammar: BaseGrammar,
    ) -> Sequence[str]:
        """Return the variable names.

        Args:
            group_name: The name of the group to which these variables shall belong.
            names: The candidate variable names.
                If empty,
                return the names of all the variables in the group of interest.
            grammar: The grammar defining the variables available in the group.

        Returns:
            The variable names.

        Raises:
            ValueError: When a variable name is not defined in the grammar.
        """
        if names:
            wrong_names = set(names).difference(grammar)
            if wrong_names:
                msg = (
                    f"{sorted(wrong_names)} are not names of {group_name} "
                    f"in the discipline {self.discipline.name}; "
                    f"expected names among {sorted(grammar)}."
                )
                raise ValueError(msg)

            return names

        return list(grammar)
