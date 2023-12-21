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

from enum import auto
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import ones
from numpy.random import default_rng
from scipy.sparse import rand as sp_rand
from strenum import LowercaseStrEnum

from gemseo import SEED
from gemseo.core.derivatives.jacobian_operator import JacobianOperator
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Sequence


class LinearDiscipline(MDODiscipline):
    """A discipline that computes random outputs from inputs.

    The output are computed by a product with a random matrix and the inputs. The inputs
    and outputs names are specified by the user. The size of inputs and outputs can be
    specified.
    """

    DEFAULT_MATRIX_DENSITY: ClassVar[float] = 0.1

    class MatrixFormat(LowercaseStrEnum):
        """The format of the Jacobian matrix.

        DENSE corresponds to numpy.ndarray. CSC, CSR, LIL and DOK correspond to sparse
        format from scipy.sparse.
        """

        DENSE = auto()
        CSC = auto()
        CSR = auto()
        LIL = auto()
        DOK = auto()

    def __init__(
        self,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        inputs_size: int = 1,
        outputs_size: int = 1,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        matrix_format: MatrixFormat = MatrixFormat.DENSE,
        matrix_density: float = DEFAULT_MATRIX_DENSITY,
        matrix_free_jacobian: bool = False,
    ) -> None:
        """
        Args:
            name: The discipline name.
            input_names: The input data names.
            output_names: The output data names.
            inputs_size: The size of input data vectors,
                each input data is of shape (inputs_size,).
            outputs_size: The size of output data vectors,
                each output data is of shape (outputs_size,).
            grammar_type: The type of grammars.
            matrix_format: The format of the Jacobian matrix.
            matrix_density: The percentage of non-zero elements when the matrix is
                sparse.
            matrix_free_jacobian: Whether the Jacobians are casted as linear operators.

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

        self.input_grammar.update_from_names(input_names)
        self.output_grammar.update_from_names(output_names)

        self.size_in = len(input_names) * inputs_size
        self.size_out = len(output_names) * outputs_size

        self.inputs_size = inputs_size
        self.outputs_size = outputs_size

        self.matrix_free_jacobian = matrix_free_jacobian

        if matrix_format == self.MatrixFormat.DENSE:
            self.mat = (
                default_rng(SEED).random((self.size_out, self.size_in)) / self.size_in
            )
        else:
            self.mat = (
                sp_rand(
                    self.size_out,
                    self.size_in,
                    density=matrix_density,
                    format=matrix_format,
                )
                / self.size_in
            )

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

        if self.matrix_free_jacobian:
            for output_name in self.output_names:
                for input_name in self.input_names:
                    jac = self.jac[output_name][input_name]

                    operator = JacobianOperator(
                        shape=jac.shape,
                        dtype=jac.dtype,
                    )

                    def matvec(x, matrix=jac):
                        return matrix @ x

                    def rmatvec(x, matrix=jac):
                        return matrix.T @ x

                    operator._matvec = matvec
                    operator._rmatvec = rmatvec

                    self.jac[output_name][input_name] = operator
