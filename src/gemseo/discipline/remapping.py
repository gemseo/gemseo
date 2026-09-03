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
#     Matthias De Lozzo
"""A discipline whose inputs and outputs map to those of another."""

from __future__ import annotations

from collections.abc import Iterable
from functools import singledispatchmethod
from typing import TYPE_CHECKING
from typing import Final
from typing import NoReturn

from gemseo.core.discipline import Discipline
from gemseo.util.constant import READ_ONLY_EMPTY_DICT
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.core.grammar.base import BaseGrammar
    from gemseo.util.typing import StrKeyMapping

Indices = tuple[str, int | Iterable[int]]
NameMapping = dict[str, str | Indices]
FormattedNameMapping = dict[str, tuple[str, slice | Iterable[int]]]

_FULL_SLICE: Final[slice] = slice(None)
"""The indices of a variable mapped as a whole."""


class RemappingDiscipline(Discipline):
    """A discipline whose inputs and outputs map to those of another.

    An input or output name mapping looks like
    `{"new_x": "x", "new_y": ("y", components)}`
    where the variable `"new_x"` corresponds to the original variable `"x"`
    and the variable `"new_y"` corresponds to some `components`
    of the original variable `"y"`.
    `components` can be an integer `i` (the `i`-th component of `y`),
    a sequence of integers `[i, j, k]`
    (the `i`-th, `j`-th and `k`-th components of `y`)
    or an iterable of integers `range(i, j+1)`
    (from the `i`-th to the `j`-th components of `y`).
    """

    default_grammar_type = Discipline.GrammarType.SIMPLER

    _input_mapping: FormattedNameMapping
    """The input names of this discipline to those of the original discipline."""

    _output_mapping: FormattedNameMapping
    """The output names of this discipline to those of the original discipline."""

    _default_original_input_data: dict[str, ndarray]
    """The default values to fill component-wise with the input data of this discipline.

    These templates are required for the input variables of the original discipline
    that are mapped component-wise only;
    they are copied at each execution
    and the components that are not mapped keep their default values.
    """

    def __init__(
        self,
        discipline: Discipline,
        input_mapping: NameMapping = READ_ONLY_EMPTY_DICT,
        output_mapping: NameMapping = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """
        Args:
            discipline: The original discipline
                for which each input variable mapped component-wise
                must have a default value as a NumPy array.
            input_mapping: The input names to the original input names.
            output_mapping: The output names to the original output names.

        Raises:
            ValueError: When an input variable of the original discipline
                is mapped component-wise but has no default value.
        """  # noqa: D205, D212, D415
        self._discipline = discipline
        original_input_grammar = discipline.io.input_grammar
        original_output_grammar = discipline.io.output_grammar
        input_mapping = input_mapping or {n: n for n in original_input_grammar}
        output_mapping = output_mapping or {n: n for n in original_output_grammar}
        self._input_mapping = self.__format_mapping(
            input_mapping, original_input_grammar
        )
        self._output_mapping = self.__format_mapping(
            output_mapping, original_output_grammar
        )
        # Only the original input variables mapped component-wise
        # require a default value,
        # used as a template of which the mapped components are overwritten
        # while the other ones keep their default values.
        original_defaults = original_input_grammar.defaults
        component_mapped_names = {
            name
            for name, indices in self._input_mapping.values()
            if indices != _FULL_SLICE
        }
        names_wo_default = component_mapped_names - original_defaults.keys()
        if names_wo_default:
            msg = (
                "The input variables of the original discipline "
                "mapped component-wise must have default values; "
                f"the following ones have no default value: "
                f"{pretty_repr(names_wo_default)}."
            )
            raise ValueError(msg)

        self._default_original_input_data = {
            name: original_defaults[name] for name in component_mapped_names
        }
        super().__init__(name=self._discipline.name)
        self.io.input_grammar.update_from_names(input_mapping.keys())
        self.io.output_grammar.update_from_names(output_mapping.keys())
        self.io.input_grammar.defaults = self.__convert_from_origin(
            original_defaults,
            {
                new_name: (name, indices)
                for new_name, (name, indices) in self._input_mapping.items()
                if name in original_defaults
            },
        )
        self.add_differentiated_inputs(
            self.__get_new_data_names(
                discipline._differentiated_input_names, self._input_mapping
            )
        )
        self.add_differentiated_outputs(
            self.__get_new_data_names(
                discipline._differentiated_output_names, self._output_mapping
            )
        )
        self.linearization_mode = discipline.linearization_mode

    @property
    def original_discipline(self) -> Discipline:
        """The original discipline."""
        return self._discipline

    @singledispatchmethod
    @staticmethod
    def __cast_mapping_value(value) -> NoReturn:
        """Cast a value of a mapping.

        Args:
            value: The value to be cast.

        Returns:
            The cast value.

        Raises:
            ValueError: When the value is neither a string nor a tuple
                with a string as first component and an integer or iterable of integers
                as second one.
        """
        msg = (
            "The values of a name mapping should be either a str or a tuple[str, Any]."
        )
        raise ValueError(msg)

    @staticmethod
    @__cast_mapping_value.register
    def _(value: str):  # -> tuple[str, slice]:
        return value, slice(None)

    @staticmethod
    @__cast_mapping_value.register
    def _(value: tuple):  # -> tuple[str, slice | Iterable[int]]:
        name, indices = value
        if isinstance(indices, int):
            return name, slice(indices, indices + 1)

        return value

    @classmethod
    def __format_mapping(
        cls, mapping: NameMapping, grammar: BaseGrammar
    ) -> FormattedNameMapping:
        """Format a mapping as `{"current_name": ("original_name", components)}`.

        Args:
            mapping: The user mapping.
            grammar: The grammar.

        Returns:
            The formatted mapping.
        """
        return {k: cls.__cast_mapping_value(v) for k, v in mapping.items()}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self._discipline.execute(self.__convert_to_origin(self.io.get_input_data()))
        return self.__convert_from_origin(
            self._discipline.io.get_output_data(), self._output_mapping
        )

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        input_mapping = self._input_mapping
        original_input_names = {input_mapping[n][0] for n in input_names}
        output_mapping = self._output_mapping
        original_output_names = {output_mapping[n][0] for n in output_names}
        self._discipline._compute_jacobian(
            input_names=original_input_names, output_names=original_output_names
        )
        original_jac = self._discipline.jac
        self.jac = {}
        for new_o_name, (o_name, o_args) in self._output_mapping.items():
            self.jac[new_o_name] = {}
            for new_i_name, (i_name, i_args) in self._input_mapping.items():
                jac = original_jac[o_name][i_name]
                self.jac[new_o_name][new_i_name] = jac[o_args, i_args]

    @staticmethod
    def __convert_from_origin(
        original_data: StrKeyMapping,
        name_mapping: FormattedNameMapping,
    ) -> StrKeyMapping:
        """Convert original data to the current format.

        Args:
            original_data: The original data
                mapping the original names to the corresponding values.
            name_mapping: The current names mapping to the original ones.

        Returns:
            The current data mapping the current names to the corresponding values.
        """
        return {
            new_name: original_data[original_name]
            if args == _FULL_SLICE
            else original_data[original_name][args]
            for new_name, (original_name, args) in name_mapping.items()
        }

    def __convert_to_origin(self, input_data: StrKeyMapping) -> StrKeyMapping:
        """Convert current input data to the original format.

        Args:
            input_data: The current input data
                mapping the current input names to the corresponding values.

        Returns:
            The original input data
            mapping the original input names to the corresponding values.
        """
        original_input_data = {
            name: value.copy()
            for name, value in self._default_original_input_data.items()
        }
        for new_name, value in input_data.items():
            original_name, args = self._input_mapping[new_name]
            if args == _FULL_SLICE:
                original_input_data[original_name] = value
            else:
                original_input_data[original_name][args] = value
        return original_input_data

    @staticmethod
    def __get_new_data_names(
        data_names: Iterable[str], name_mapping: FormattedNameMapping
    ) -> list[str]:
        """Return new data names from data names based on a name mapping.

        Args:
            data_names: The data names.
            name_mapping: The name mapping.

        Returns:
            The new data names.
        """
        return [
            new_name
            for new_name, (name, _) in name_mapping.items()
            if name in data_names
        ]
