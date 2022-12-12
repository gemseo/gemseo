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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author : Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Gradient approximation by complex step."""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Sequence

from numpy import complex128
from numpy import finfo
from numpy import ndarray
from numpy import where
from numpy import zeros
from numpy.linalg import norm

from gemseo.algos.design_space import DesignSpace
from gemseo.core.derivatives.derivation_modes import COMPLEX_STEP
from gemseo.core.parallel_execution import ParallelExecution
from gemseo.utils.derivatives.gradient_approximator import GradientApproximator

EPSILON = finfo(float).eps


class ComplexStep(GradientApproximator):
    r"""Complex step approximator, performing a second-order gradient calculation.

    Enable a much lower step than real finite differences,
    typically 1e-30,
    since there is no cancellation error due to a difference calculation.

    .. math::

        \frac{df(x)}{dx} \approx Im\left( \frac{f(x+j*\\delta x)}{\\delta x} \right)

    See
    Martins, Joaquim RRA, Peter Sturdza, and Juan J. Alonso.
    "The complex-step derivative approximation."
    ACM Transactions on Mathematical Software (TOMS) 29.3 (2003): 245-262.
    """

    ALIAS = COMPLEX_STEP

    def __init__(
        self,
        f_pointer: Callable[[ndarray], ndarray],
        step: complex = 1e-20,
        parallel: bool = False,
        design_space: DesignSpace | None = None,
        normalize: bool = True,
        **parallel_args: int | bool | float,
    ) -> None:
        if design_space is not None:
            design_space.to_complex()
        super().__init__(
            f_pointer,
            step=step,
            parallel=parallel,
            design_space=design_space,
            normalize=True,
            **parallel_args,
        )

    @GradientApproximator.step.setter
    def step(self, value):
        if value.imag != 0:
            self._step = value.imag
        else:
            self._step = value

    def f_gradient(
        self,
        x_vect: ndarray,
        step: complex | None = None,
        x_indices: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> ndarray:
        if norm(x_vect.imag) != 0.0:
            raise ValueError(
                "Impossible to check the gradient at a complex "
                "point using the complex step method."
            )
        return super().f_gradient(x_vect, step=step, x_indices=x_indices, **kwargs)

    def _compute_parallel_grad(
        self,
        input_values: ndarray,
        n_perturbations: int,
        input_perturbations: ndarray,
        step: float,
        **kwargs: Any,
    ) -> ndarray:
        def func_noargs(
            f_input_values: ndarray,
        ) -> ndarray:
            """Call the function without explicitly passed arguments.

            Args:
                f_input_values: The input value.

            Return:
                The value of the function output.
            """
            return self.f_pointer(f_input_values, **kwargs)

        functions = [func_noargs] * n_perturbations
        parallel_execution = ParallelExecution(functions, **self._par_args)

        perturbated_inputs = [
            input_values + input_perturbations[:, perturbation_index]
            for perturbation_index in range(n_perturbations)
        ]
        perturbated_outputs = parallel_execution.execute(perturbated_inputs)

        gradient = []
        for perturbation_index in range(n_perturbations):
            gradient.append(
                perturbated_outputs[perturbation_index].imag
                / input_perturbations[perturbation_index, perturbation_index].imag
            )

        return gradient

    def _compute_grad(
        self,
        input_values: ndarray,
        n_perturbations: int,
        input_perturbations: ndarray,
        step: float,
        **kwargs: Any,
    ) -> ndarray:
        gradient = []
        for perturbation_index in range(n_perturbations):
            perturbated_input = (
                input_values + input_perturbations[:, perturbation_index]
            )
            perturbated_output = self.f_pointer(perturbated_input, **kwargs)
            gradient.append(
                perturbated_output.imag
                / input_perturbations[perturbation_index, perturbation_index].imag
            )

        return gradient

    def _generate_perturbations(
        self,
        input_values: ndarray,
        input_indices: list[int],
        step: float,
    ) -> tuple[ndarray, float | ndarray]:
        input_dimension = len(input_values)
        n_indices = len(input_indices)
        input_perturbations = zeros((input_dimension, n_indices), dtype=complex128)
        x_nnz = where(input_values == 0.0, 1.0, input_values)[input_indices]
        input_perturbations[input_indices, range(n_indices)] = 1j * x_nnz * step
        return input_perturbations, step
