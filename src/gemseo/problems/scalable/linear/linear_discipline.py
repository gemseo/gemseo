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
"""Dummy linear discipline."""
from __future__ import annotations

from typing import Sequence

from numpy import ones
from numpy.random import rand

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays


class LinearDiscipline(MDODiscipline):
    """A discipline that computes random outputs from inputs.

    The output are computed by a product with a random matrix and the inputs. The inputs
    and output names are specified by the user. The size of inputs and outputs can be
    specified.
    """

    def __init__(
        self,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        inputs_size: int = 1,
        outputs_size: int = 1,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
    ) -> None:
        """
        Args:
            name: The discipline name.
            input_names: The input data names
            output_names: The output data names.
            inputs_size: The size of input data vectors,
                each input data is of shape (inputs_size,).
            outputs_size: The size of output data vectors,
                each output data is of shape (outputs_size,).
            grammar_type: The type of grammars.

        Raises:
            ValueError: if ``input_names`` or ``output_names`` are empty.
        """  # noqa: D205, D212, D415
        if not input_names:
            raise ValueError("input_names must not be empty.")
        if not output_names:
            raise ValueError("output_names must not be empty.")
        super().__init__(name, grammar_type=grammar_type)
        self.input_names = input_names
        self.output_names = output_names

        self.input_grammar.update(input_names)
        self.output_grammar.update(output_names)

        self.size_in = len(input_names) * inputs_size
        self.size_out = len(output_names) * outputs_size

        self.inputs_size = inputs_size
        self.outputs_size = outputs_size
        self.mat = rand(self.size_out, self.size_in) / self.size_in

        self.__sizes_d = {k: self.inputs_size for k in self.input_names}
        self.__sizes_d.update({k: self.outputs_size for k in self.output_names})

        self.default_inputs = {k: 0.5 * ones(inputs_size) for k in input_names}

    def _run(self) -> None:
        input_data = concatenate_dict_of_arrays_to_array(
            self.local_data, self.input_names
        )
        output_data = self.mat.dot(input_data)
        self.local_data.update(
            split_array_to_dict_of_arrays(
                output_data, self.__sizes_d, self.output_names
            )
        )

    def _compute_jacobian(
        self,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        self.jac = split_array_to_dict_of_arrays(
            self.mat, self.__sizes_d, self.output_names, self.input_names
        )
