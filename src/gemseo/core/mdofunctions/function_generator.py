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
"""A class to creat MDOFunctions from MDODisciplines."""
from __future__ import annotations

import logging
from numbers import Number
from typing import Callable
from typing import Mapping
from typing import Sequence
from typing import TYPE_CHECKING
from typing import Union

from numpy import ndarray

from gemseo.core.discipline import MDODiscipline
from gemseo.core.mdofunctions.make_function import MakeFunction

if TYPE_CHECKING:
    from gemseo.core.mdofunctions.mdo_function import MDOFunction

LOGGER = logging.getLogger(__name__)

OperandType = Union[ndarray, Number]
OperatorType = Callable[[OperandType, OperandType], OperandType]


class MDOFunctionGenerator:
    """Generator of :class:`.MDOFunction` objects from a :class:`.MDODiscipline`.

    It creates a :class:`.MDOFunction`
    evaluating some of the outputs of the discipline
    from some of its

    It uses closures to generate functions instances from a discipline execution.
    """

    def __init__(
        self,
        discipline: MDODiscipline,
    ) -> None:
        """
        Args:
            discipline: The discipline from which the generator builds the functions.
        """  # noqa: D205, D212, D415
        self.discipline = discipline

    def get_function(
        self,
        input_names: Sequence[str],
        output_names: Sequence[str],
        default_inputs: Mapping[str, ndarray] | None = None,
        differentiable: bool = True,
    ) -> MDOFunction:
        """Build a function from a discipline input and output lists.

        Args:
            input_names: The names of the inputs of the discipline
                to be inputs of the function.
            output_names: The names of outputs of the discipline
                to be returned by the function.
            default_inputs: The default values of the inputs.
                If None,
                use the default values of the inputs
                specified by the discipline.
            differentiable: If True, then inputs and outputs are added
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
                "Some elements of {} are not inputs of the discipline {}; "
                "available inputs are: {}.".format(
                    input_names,
                    self.discipline.name,
                    self.discipline.get_input_data_names(),
                )
            )

        if not self.discipline.is_all_outputs_existing(output_names):
            raise ValueError(
                "Some elements of {} are not outputs of the discipline {}; "
                "available outputs are: {}.".format(
                    output_names,
                    self.discipline.name,
                    ", ".join(self.discipline.get_output_data_names()),
                )
            )

        # adds inputs and outputs to the list of variables to be
        # differentiated
        if differentiable:
            self.discipline.add_differentiated_inputs(input_names)
            self.discipline.add_differentiated_outputs(output_names)

        return MakeFunction(input_names, output_names, default_inputs, self)
