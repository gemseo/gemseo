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

from typing import Dict
from typing import Iterable
from typing import Tuple
from typing import Union

from numpy import empty
from numpy import ndarray

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.python_compatibility import singledispatchmethod

Data = Dict[str, ndarray]
Indices = Tuple[str, Union[int, Iterable[int]]]
NameMapping = Dict[str, Union[str, Indices]]


class RemappingDiscipline(MDODiscipline):
    """A discipline whose inputs and outputs map to those of another.

    An input or output name mapping looks like
    ``{"new_x": "x", "new_y": ("y", components)}``
    where the variable ``"new_x"`` corresponds to the original variable ``"x"``
    and the variable ``"new_y"`` corresponds to some ``components``
    of the original variable ``"y"``.
    ``components`` can be an integer ``i`` (the ``i``-th component of ``y``),
    a sequence of integers ``[i, j, k]``
    (the ``i``-th, ``j``-th and ``k``-th components of ``y``)
    or an iterable of integers ``range(i, j+1)``
    (from the ``i``-th to the ``j``-th components of ``y``).
    """

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "_discipline",
        "_empty_original_input_data",
        "_input_mapping",
        "_output_mapping",
    )

    def __init__(
        self,
        discipline: MDODiscipline,
        input_mapping: NameMapping,
        output_mapping: NameMapping,
    ) -> None:
        """..
        Args:
            discipline: The original discipline.
            input_mapping: The input names to the original input names.
            output_mapping: The output names to the original output names.

        Raises:
            ValueError: When the original discipline has no default input values.
        """  # noqa: D205, D212, D415
        if not discipline.default_inputs:
            raise ValueError("The original discipline has no default input values.")

        self._discipline = discipline
        self._empty_original_input_data = {
            k: empty(v.shape, dtype=v.dtype)
            for k, v in discipline.default_inputs.items()
        }
        self._input_mapping = self.__format_mapping(input_mapping)
        self._output_mapping = self.__format_mapping(output_mapping)
        super().__init__(name=self._discipline.name)
        self.input_grammar.update(self._input_mapping.keys())
        self.output_grammar.update(self._output_mapping.keys())
        self.default_inputs = self.__convert_from_origin(
            discipline.default_inputs, self._input_mapping
        )

    @property
    def original_discipline(self) -> MDODiscipline:
        """The original discipline."""
        return self._discipline

    @singledispatchmethod
    @staticmethod
    def __cast_mapping_value(value) -> slice | Iterable[int]:
        """Cast a value of a mapping.

        Args:
            value: The value to be cast.

        Returns:
            The casted value.

        Raises:
            ValueError: When the value is neither a string nor a tuple
                with a string as first component and an integer or iterable of integers
                as second one.
        """
        raise ValueError(
            "The values of a name mapping should be either a str or a tuple[str, Any]."
        )

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
    def __format_mapping(cls, mapping: NameMapping) -> dict[str, slice | Iterable[int]]:
        """Format a mapping as ``{"current_name": ("original_name", components)}``.

        Args:
            mapping: The user mapping.

        Returns:
            The formatted mapping.
        """
        return {k: cls.__cast_mapping_value(v) for k, v in mapping.items()}

    def _run(self) -> None:
        self._discipline.execute(self.__convert_to_origin(self.get_input_data()))
        self.local_data.update(
            self.__convert_from_origin(
                self._discipline.get_output_data(), self._output_mapping
            )
        )

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._discipline._compute_jacobian(inputs=inputs, outputs=outputs)
        original_jac = self._discipline.jac
        self.jac = {}
        for new_o_name, (o_name, o_args) in self._output_mapping.items():
            self.jac[new_o_name] = {}
            for new_i_name, (i_name, i_args) in self._input_mapping.items():
                jac = original_jac[o_name][i_name]
                self.jac[new_o_name][new_i_name] = jac[o_args, i_args]

    @staticmethod
    def __convert_from_origin(original_data: Data, name_mapping: NameMapping) -> Data:
        """Convert original data to the current format.

        Args:
            original_data: The original data
                mapping the original names to the corresponding values.
            name_mapping: The current names mapping to the original ones.

        Returns:
            The current data
            mapping the current names to the corresponding values.
        """
        return {
            new_name: original_data[original_name][args]
            for new_name, (original_name, args) in name_mapping.items()
        }

    def __convert_to_origin(self, input_data: Data) -> Data:
        """Convert current input data to the original format.

        Args:
            input_data: The current input data
                mapping the current input names to the corresponding values.

        Returns:
            The original input data
            mapping the original input names to the corresponding values.
        """
        original_input_data = self._empty_original_input_data.copy()
        for new_name, value in input_data.items():
            original_name, args = self._input_mapping[new_name]
            original_input_data[original_name][args] = value
        return original_input_data
