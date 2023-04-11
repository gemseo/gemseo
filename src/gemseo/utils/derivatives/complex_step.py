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
from typing import Final
from typing import Sequence

from numpy import complex128
from numpy import ndarray
from numpy import where
from numpy import zeros
from numpy.linalg import norm

from gemseo.algos.design_space import DesignSpace
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.gradient_approximator import GradientApproximator


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

    _APPROXIMATION_MODE = ApproximationMode.COMPLEX_STEP

    __DEFAULT_STEP: Final[complex] = 1e-20
    """The default value for the step."""

    def __init__(  # noqa:D107
        self,
        f_pointer: Callable[[ndarray], ndarray],
        step: complex | None = None,
        design_space: DesignSpace | None = None,
        normalize: bool = True,
        parallel: bool = False,
        **parallel_args: int | bool | float,
    ) -> None:
        if design_space is not None:
            design_space.to_complex()
        super().__init__(
            f_pointer,
            step=self.__DEFAULT_STEP if step is None else step,
            parallel=parallel,
            design_space=design_space,
            normalize=True,
            **parallel_args,
        )

    @GradientApproximator.step.setter
    def step(self, value) -> None:  # noqa:D102
        if value.imag != 0:
            self._step = value.imag
        else:
            self._step = value

    def f_gradient(  # noqa:D102
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
        self._function_kwargs = kwargs
        functions = [self._wrap_function] * n_perturbations
        parallel_execution = CallableParallelExecution(functions, **self._parallel_args)

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
