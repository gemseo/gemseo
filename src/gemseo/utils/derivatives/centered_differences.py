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
"""Gradient approximation by centered differences."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import argmax
from numpy import concatenate
from numpy import full
from numpy import tile
from numpy import where
from numpy import zeros
from numpy.linalg import norm

from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.base_gradient_approximator import BaseGradientApproximator
from gemseo.utils.derivatives.error_estimators import EPSILON
from gemseo.utils.derivatives.error_estimators import compute_best_step

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class CenteredDifferences(BaseGradientApproximator):
    r"""Centered differences approximator.

    .. math::

        \frac{df(x)}{dx}\approx\frac{f(x+\delta x)-f(x-\delta x)}{2\delta x}
    """

    _APPROXIMATION_MODE = ApproximationMode.CENTERED_DIFFERENCES

    _DEFAULT_STEP: ClassVar[float] = 1.0e-6

    def _compute_parallel_grad(
        self,
        input_values: RealArray,
        input_perturbations: RealArray,
        step: float | RealArray,
        **kwargs: Any,
    ) -> list[RealArray]:
        input_perturbations = input_perturbations.T
        n_perturbations = len(input_perturbations)
        self._function_kwargs = kwargs
        parallel_execution = CallableParallelExecution(
            [self._wrap_function] * n_perturbations, **self._parallel_args
        )
        output_perturbations = parallel_execution.execute(input_perturbations)

        n_perturbations_ = int(n_perturbations / 2)
        return [
            ((output_plus - output_minus) / norm(input_plus - input_minus)).real
            for input_plus, output_plus, input_minus, output_minus in zip(
                input_perturbations[:n_perturbations_],
                output_perturbations[:n_perturbations_],
                input_perturbations[n_perturbations_ : 2 * n_perturbations_],
                output_perturbations[n_perturbations_ : 2 * n_perturbations_],
            )
        ]

    def _compute_grad(
        self,
        input_values: RealArray,
        input_perturbations: RealArray,
        step: float | RealArray,
        **kwargs: Any,
    ) -> list[RealArray]:
        input_perturbations = input_perturbations.T
        n_perturbations_ = int(len(input_perturbations) / 2)
        f = self.f_pointer
        return [
            (
                (f(input_plus, **kwargs) - f(input_minus, **kwargs))
                / norm(input_plus - input_minus)
            ).real
            for input_plus, input_minus in zip(
                input_perturbations[:n_perturbations_],
                input_perturbations[n_perturbations_ : 2 * n_perturbations_],
            )
        ]

    def _get_opt_step(
        self,
        f_p: RealArray,
        f_0: RealArray,
        f_m: RealArray,
        numerical_error: float = EPSILON,
    ) -> tuple[RealArray | float, RealArray]:
        r"""Compute the optimal step of a function.

        This function may be a vector function.
        In this case, take the worst case.

        Args:
            f_p: The value of the function :math:`f`
                 at the next step :math:`x+\\delta_x`.
            f_0: The value of the function :math:`f`
                 at the current step :math:`x`.
            f_m: The value of the function :math:`f`
                 at the previous step :math:`x-\\delta_x`.
            numerical_error: The numerical error
                associated to the calculation of :math:`f`.
                By default, Machine epsilon (appx 1e-16),
                but can be higher.
                when the calculation of :math:`f` requires a numerical resolution.

        Returns:
            The errors.
            The optimal steps.
        """
        n_out = f_p.size
        if n_out == 1:
            t_e, c_e, opt_step = compute_best_step(
                f_p, f_0, f_m, self.step, epsilon_mach=numerical_error
            )
            return 0.0 if t_e is None else t_e + c_e, opt_step

        errors = zeros(n_out)
        opt_steps = zeros(n_out)
        for i in range(n_out):
            t_e, c_e, opt_steps[i] = compute_best_step(
                f_p[i], f_0[i], f_m[i], self.step, epsilon_mach=numerical_error
            )
            errors[i] = 0.0 if t_e is None else t_e + c_e

        max_i = argmax(errors)
        return errors[max_i], opt_steps[max_i]

    def compute_optimal_step(
        self,
        x_vect: RealArray,
        numerical_error: float = EPSILON,
        **kwargs: Any,
    ) -> tuple[RealArray, RealArray]:
        r"""Compute the gradient by real step.

        Args:
            x_vect: The input vector.
            numerical_error: The numerical error
                associated to the calculation of :math:`f`.
                By default, machine epsilon (appx 1e-16),
                but can be higher.
                when the calculation of :math:`f` requires a numerical resolution.
            **kwargs: The additional arguments passed to the function.

        Returns:
            The optimal steps.
            The errors.
        """
        n_dim = len(x_vect)
        x_p_arr = self.generate_perturbations(n_dim, x_vect)[0]
        x_m_arr = self.generate_perturbations(n_dim, x_vect, step=-self.step)[0]
        opt_steps = full(n_dim, self.step)
        errors = zeros(n_dim)
        comp_step = self._get_opt_step
        if self._parallel:
            self._function_kwargs = kwargs
            workers = [self._wrap_function] * (n_dim * 2 + 1)
            execution = CallableParallelExecution(workers, **self._parallel_args)
            outputs = execution.execute([
                x_vect,
                *[x_p_arr[:, i] for i in range(n_dim)],
                *[x_m_arr[:, i] for i in range(n_dim)],
            ])

            f_0 = outputs[0]
            for i in range(n_dim):
                errs, opt_step = comp_step(
                    outputs[i + 1],
                    f_0,
                    outputs[n_dim + i + 1],
                    numerical_error=numerical_error,
                )
                errors[i] = errs
                opt_steps[i] = opt_step
        else:
            compute_output = self.f_pointer
            f_0 = compute_output(x_vect, **kwargs)
            for i in range(n_dim):
                errors[i], opt_steps[i] = comp_step(
                    compute_output(x_p_arr[:, i], **kwargs),
                    f_0,
                    compute_output(x_m_arr[:, i], **kwargs),
                    numerical_error=numerical_error,
                )

        self.step = opt_steps
        return opt_steps, errors

    def _generate_perturbations(
        self,
        input_values: RealArray,
        input_indices: list[int],
        step: float,
    ) -> tuple[RealArray, RealArray | float]:
        input_dimension = len(input_values)
        n_indices = len(input_indices)
        input_perturbations = (
            tile(input_values, 2 * n_indices)
            .reshape((2 * n_indices, input_dimension))
            .T
        )
        if self._design_space is None:
            input_perturbations[input_indices, range(n_indices)] += step
            input_perturbations[input_indices, range(n_indices, 2 * n_indices)] -= step
            return input_perturbations, step

        lower_bounds = self._design_space.get_lower_bounds()
        upper_bounds = self._design_space.get_upper_bounds()
        if self._normalize:
            normalize_vect = self._design_space.normalize_vect
            lower_bounds = normalize_vect(lower_bounds)
            upper_bounds = normalize_vect(upper_bounds)

        steps_plus = where(
            input_perturbations[input_indices, range(n_indices)] >= upper_bounds,
            0,
            step,
        )
        input_perturbations[input_indices, range(n_indices)] += steps_plus
        steps_minus = where(
            input_perturbations[input_indices, range(n_indices, 2 * n_indices)]
            <= lower_bounds,
            0,
            -step,
        )
        steps = concatenate([steps_plus, steps_minus], axis=-1)
        input_perturbations[input_indices, range(n_indices, 2 * n_indices)] += (
            steps_minus
        )
        return input_perturbations, steps
