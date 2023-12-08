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
"""Gradient approximation by finite differences."""

from __future__ import annotations

from typing import Any
from typing import ClassVar

from numpy import argmax
from numpy import full
from numpy import ndarray
from numpy import tile
from numpy import where
from numpy import zeros

from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.derivatives.error_estimators import EPSILON
from gemseo.utils.derivatives.error_estimators import compute_best_step
from gemseo.utils.derivatives.gradient_approximator import GradientApproximator


class FirstOrderFD(GradientApproximator):
    r"""First-order finite differences approximator.

    .. math::

        \frac{df(x)}{dx}\approx\frac{f(x+\\delta x)-f(x)}{\\delta x}
    """

    _APPROXIMATION_MODE = ApproximationMode.FINITE_DIFFERENCES

    _DEFAULT_STEP: ClassVar[float] = 1.0e-6

    def _compute_parallel_grad(
        self,
        input_values: ndarray,
        n_perturbations: int,
        input_perturbations: ndarray,
        step: float | ndarray,
        **kwargs: Any,
    ) -> ndarray:
        if step is None:
            step = self.step

        if not isinstance(step, ndarray):
            step = full(n_perturbations, step)

        self._function_kwargs = kwargs
        functions = [self._wrap_function] * (n_perturbations + 1)
        parallel_execution = CallableParallelExecution(functions, **self._parallel_args)

        perturbated_inputs = [
            input_perturbations[:, perturbation_index]
            for perturbation_index in range(n_perturbations)
        ]
        initial_and_perturbated_outputs = parallel_execution.execute([
            input_values,
            *perturbated_inputs,
        ])

        gradient = []
        initial_output = initial_and_perturbated_outputs[0]
        for perturbation_index in range(n_perturbations):
            perturbated_output = initial_and_perturbated_outputs[perturbation_index + 1]
            g_approx = (perturbated_output - initial_output) / step[perturbation_index]
            gradient.append(g_approx.real)

        return gradient

    def _compute_grad(
        self,
        input_values: ndarray,
        n_perturbations: int,
        input_perturbations: ndarray,
        step: float | ndarray,
        **kwargs: Any,
    ) -> ndarray:
        if step is None:
            step = self.step

        if not isinstance(step, ndarray):
            step = full(n_perturbations, step)

        gradient = []
        initial_output = self.f_pointer(input_values, **kwargs)
        for perturbation_index in range(n_perturbations):
            perturbated_output = self.f_pointer(
                input_perturbations[:, perturbation_index], **kwargs
            )
            g_approx = (perturbated_output - initial_output) / step[perturbation_index]
            gradient.append(g_approx.real)

        return gradient

    def _get_opt_step(
        self,
        f_p: ndarray,
        f_0: ndarray,
        f_m: ndarray,
        numerical_error: float = EPSILON,
    ) -> tuple[ndarray, ndarray]:
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
            error = 0.0 if t_e is None else t_e + c_e
        else:
            errors = zeros(n_out)
            opt_steps = zeros(n_out)
            for i in range(n_out):
                t_e, c_e, opt_steps[i] = compute_best_step(
                    f_p[i], f_0[i], f_m[i], self.step, epsilon_mach=numerical_error
                )
                if t_e is None:
                    errors[i] = 0.0
                else:
                    errors[i] = t_e + c_e
            max_i = argmax(errors)
            error = errors[max_i]
            opt_step = opt_steps[max_i]

        return error, opt_step

    def compute_optimal_step(
        self,
        x_vect: ndarray,
        numerical_error: float = EPSILON,
        **kwargs,
    ) -> tuple[ndarray, ndarray]:
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
        x_p_arr, _ = self.generate_perturbations(n_dim, x_vect)
        x_m_arr, _ = self.generate_perturbations(n_dim, x_vect, step=-self.step)
        opt_steps = full(n_dim, self.step)
        errors = zeros(n_dim)
        comp_step = self._get_opt_step
        if self._parallel:
            self._function_kwargs = kwargs
            functions = [self._wrap_function] * (n_dim * 2 + 1)
            parallel_execution = CallableParallelExecution(
                functions, **self._parallel_args
            )

            all_x = [x_vect] + [x_p_arr[:, i] for i in range(n_dim)]
            all_x += [x_m_arr[:, i] for i in range(n_dim)]
            outputs = parallel_execution.execute(all_x)

            f_0 = outputs[0]
            for i in range(n_dim):
                f_p = outputs[i + 1]
                f_m = outputs[n_dim + i + 1]
                errs, opt_step = comp_step(
                    f_p, f_0, f_m, numerical_error=numerical_error
                )
                errors[i] = errs
                opt_steps[i] = opt_step
        else:
            f_0 = self.f_pointer(x_vect, **kwargs)
            for i in range(n_dim):
                f_p = self.f_pointer(x_p_arr[:, i], **kwargs)
                f_m = self.f_pointer(x_m_arr[:, i], **kwargs)
                errs, opt_step = comp_step(
                    f_p, f_0, f_m, numerical_error=numerical_error
                )
                errors[i] = errs
                opt_steps[i] = opt_step
        self.step = opt_steps
        return opt_steps, errors

    def _generate_perturbations(
        self,
        input_values: ndarray,
        input_indices: list[int],
        step: float,
    ) -> tuple[ndarray, ndarray]:
        input_dimension = len(input_values)
        n_indices = len(input_indices)
        input_perturbations = (
            tile(input_values, n_indices).reshape((n_indices, input_dimension)).T
        )
        if self._design_space is None:
            input_perturbations[input_indices, range(n_indices)] += step
            return input_perturbations, step

        if self._normalize:
            upper_bounds = self._design_space.normalize_vect(
                self._design_space.get_upper_bounds()
            )
        else:
            upper_bounds = self._design_space.get_upper_bounds()

        steps = where(
            input_perturbations[input_indices, range(n_indices)] >= upper_bounds,
            -step,
            step,
        )
        input_perturbations[input_indices, range(n_indices)] += steps

        return input_perturbations, steps
