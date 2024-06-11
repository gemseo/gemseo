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
"""A class to create :class:`.MDOFunction` objects from an :class:`.MDODiscipline`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.mdofunctions.mdo_discipline_adapter import MDODisciplineAdapter
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.core.discipline import MDODiscipline
    from gemseo.core.grammars.base_grammar import BaseGrammar


class MDODisciplineAdapterGenerator:
    """A generator of discipline adapter.

    Given a discipline,
    an :class:`.MDODisciplineAdapter` computes specific outputs from specific inputs.
    """

    discipline: MDODiscipline
    """The discipline from which to generate discipline adapters."""

    __names_to_sizes: MutableMapping[str, int]
    """The names of the inputs bound to their sizes, if known."""

    def __init__(
        self,
        discipline: MDODiscipline,
        names_to_sizes: MutableMapping[str, int] = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """
        Args:
            discipline: The discipline from which to generate discipline adapters.
            names_to_sizes: The sizes of the input variables.
                If empty,
                determine them from the default inputs and local data of the discipline.
        """  # noqa: D205, D212, D415
        self.discipline = discipline
        self.__names_to_sizes = names_to_sizes or {}

    def get_function(
        self,
        input_names: Sequence[str],
        output_names: Sequence[str],
        default_inputs: Mapping[str, ndarray] = READ_ONLY_EMPTY_DICT,
        is_differentiable: bool = True,
        differentiated_input_names_substitute: Sequence[str] = (),
    ) -> MDODisciplineAdapter:
        """Build a function executing a discipline for some inputs and outputs.

        Args:
            input_names: The discipline input names defining the function input vector.
                If empty,
                use all the discipline inputs.
            output_names: The discipline output names
                defining the function output vector.
                If empty,
                use all the discipline outputs.
            default_inputs: The default values of the input variables.
                If empty,
                use the default input values of the discipline.
            is_differentiable: Whether the function is differentiable.
            differentiated_input_names_substitute: The names of the inputs
                against which to differentiate the functions.
                If empty,
                use ``input_names``.
                This argument is not used when ``is_differentiable`` is ``False``.

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
            self.discipline.input_grammar,
        )
        output_names = self.__get_names(
            "outputs",
            output_names,
            self.discipline.output_grammar,
        )
        if differentiated_input_names_substitute:
            self.__get_names(
                "inputs",
                differentiated_input_names_substitute,
                self.discipline.input_grammar,
            )
        else:
            differentiated_input_names_substitute = input_names

        if is_differentiable:
            self.discipline.add_differentiated_inputs(
                differentiated_input_names_substitute
            )
            self.discipline.add_differentiated_outputs(output_names)

        return MDODisciplineAdapter(
            input_names,
            output_names,
            default_inputs or {},
            self.discipline,
            self.__names_to_sizes,
            differentiated_input_names_substitute=differentiated_input_names_substitute,
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
            wrong_names = set(names) - grammar.names
            if wrong_names:
                msg = (
                    f"{sorted(wrong_names)} are not names of {group_name} "
                    f"in the discipline {self.discipline.name}; "
                    f"expected names among {sorted(grammar.names)}."
                )
                raise ValueError(msg)

            return names

        return list(grammar.names)
