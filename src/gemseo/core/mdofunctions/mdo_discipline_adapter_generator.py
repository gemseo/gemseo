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

import logging
from numbers import Number
from typing import TYPE_CHECKING
from typing import Callable
from typing import Union

from numpy import ndarray

from gemseo.core.mdofunctions.mdo_discipline_adapter import MDODisciplineAdapter

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from collections.abc import Sequence

    from gemseo.core.discipline import MDODiscipline

LOGGER = logging.getLogger(__name__)

OperandType = Union[ndarray, Number]
OperatorType = Callable[[OperandType, OperandType], OperandType]


class MDODisciplineAdapterGenerator:
    """Generator of :class:`.MDOFunction` objects executing an :class:`.MDODiscipline`.

    It creates an :class:`.MDODisciplineAdapter` evaluating some of the outputs of the
    discipline from some of its

    It uses closures to generate functions instances from a discipline execution.
    """

    discipline: MDODiscipline
    """The discipline from which to generate functions."""

    __names_to_sizes: MutableMapping[str, int] | None = None
    """The names of the inputs bound to their sizes, if known."""

    def __init__(
        self,
        discipline: MDODiscipline,
        names_to_sizes: MutableMapping[str, int] | None = None,
    ) -> None:
        """
        Args:
            discipline: The discipline from which the generator builds the functions.
            names_to_sizes: The sizes of the input variables.
                If ``None``, guess them from the default inputs and local data
                of the discipline :class:`.MDODiscipline`.
        """  # noqa: D205, D212, D415
        self.discipline = discipline
        self.__names_to_sizes = names_to_sizes

    def get_function(
        self,
        input_names: Sequence[str],
        output_names: Sequence[str],
        default_inputs: Mapping[str, ndarray] | None = None,
        differentiable: bool = True,
    ) -> MDODisciplineAdapter:
        """Build a function executing a discipline for some inputs and outputs.

        Args:
            input_names: The names of the inputs of the discipline
                to be inputs of the function.
            output_names: The names of outputs of the discipline
                to be returned by the function.
            default_inputs: The default values of the inputs.
                If ``None``,
                use the default values of the inputs
                specified by the discipline.
            differentiable: If ``True``, then inputs and outputs are added
                to the variables to be differentiated.

        Returns:
            The function.

        Raises:
            ValueError: If a given input (or output) name is not the name
                of an input (or output) variable of the discipline.
        """
        if isinstance(input_names, str):
            input_names = [input_names]

        if isinstance(output_names, str):
            output_names = [output_names]

        if input_names is None:
            input_names = self.discipline.get_input_data_names()
        if output_names is None:
            output_names = self.discipline.get_output_data_names()

        if not self.discipline.is_all_inputs_existing(input_names):
            raise ValueError(
                f"Some elements of {input_names} "
                f"are not inputs of the discipline {self.discipline.name}; "
                f"available inputs are: {self.discipline.get_input_data_names()}."
            )

        if not self.discipline.is_all_outputs_existing(output_names):
            raise ValueError(
                f"Some elements of {output_names} "
                f"are not outputs of the discipline {self.discipline.name}; "
                f"available outputs are: {self.discipline.get_output_data_names()}."
            )

        # adds inputs and outputs to the list of variables to be
        # differentiated
        if differentiable:
            self.discipline.add_differentiated_inputs(input_names)
            self.discipline.add_differentiated_outputs(output_names)
        return MDODisciplineAdapter(
            input_names,
            output_names,
            default_inputs,
            self.discipline,
            self.__names_to_sizes,
            linear_candidate=self.__is_linear(input_names, output_names),
        )

    def __is_linear(
        self, input_names: Sequence[str], output_names: Sequence[str]
    ) -> bool:
        """Check if the MDOFunction should be linear.

        Args:
            input_names: The names of the inputs of the discipline
                to be inputs of the function.
            output_names: The names of outputs of the discipline
                to be returned by the function.

        Returns:
            Whether the function should be linear.
        """
        input_names = set(input_names)
        for output_name in output_names:
            linear_input_names = self.discipline.linear_relationships.get(output_name)
            if linear_input_names is not None:
                if not input_names.issubset(linear_input_names):
                    return False
            else:
                return False
        return True
