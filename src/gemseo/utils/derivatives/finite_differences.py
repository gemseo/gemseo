# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from numpy import argmax, finfo, ndarray, ones, tile, where, zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.core.parallel_execution import ParallelExecution
from gemseo.utils.derivatives.gradient_approximator import GradientApproximator
from gemseo.utils.py23_compat import xrange

EPSILON = finfo(float).eps
LOGGER = logging.getLogger(__name__)


class FirstOrderFD(GradientApproximator):
    """First-order finite differences approximator.

    .. math::

        \frac{df(x)}{dx}\approx\frac{f(x+\\delta x)-f(x)}{\\delta x}
    """

    ALIAS = "finite_differences"

    def __init__(
        self,
        f_pointer,  # type: Callable[[ndarray],ndarray]
        step=1e-6,  # type: Union[float,ndarray]
        parallel=False,  # type: bool
        design_space=None,  # type: Optional[DesignSpace]
        normalize=True,  # type: bool
        **parallel_args  # type: Union[int,bool,float]
    ):  # type: (...) -> None
        super(FirstOrderFD, self).__init__(
            f_pointer,
            step=step,
            parallel=parallel,
            design_space=design_space,
            normalize=normalize,
            **parallel_args
        )

    def f_gradient(
        self,
        x_vect,  # type: ndarray
        step=None,  # type: Optional[float,ndarray]
        x_indices=None,  # type: Optional[Sequence[int]]
        **kwargs  # type: Any
    ):  # type: (...) -> ndarray
        return super(FirstOrderFD, self).f_gradient(
            x_vect, step=step, x_indices=x_indices, **kwargs
        )

    def _compute_parallel_grad(
        self,
        input_values,  # type: ndarray
        n_perturbations,  # type: int
        input_perturbations,  # type: ndarray
        step,  # type: Union[float,ndarray]
        **kwargs  # type: Any
    ):  # type: (...) -> ndarray
        if step is None:
            step = self.step

        if not isinstance(step, ndarray):
            step = step * ones(n_perturbations)

        def func_noargs(
            f_input_values,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Call the function without explicitly passed arguments.

            Args:
                f_input_values: The input value.

            Return:
                The value of the function output.
            """
            return self.f_pointer(f_input_values, **kwargs)

        functions = [func_noargs] * (n_perturbations + 1)
        parallel_execution = ParallelExecution(functions, **self._par_args)

        perturbated_inputs = [
            input_perturbations[:, perturbation_index]
            for perturbation_index in xrange(n_perturbations)
        ]
        initial_and_perturbated_outputs = parallel_execution.execute(
            [input_values] + perturbated_inputs
        )

        gradient = []
        initial_output = initial_and_perturbated_outputs[0]
        for perturbation_index in xrange(n_perturbations):
            perturbated_output = initial_and_perturbated_outputs[perturbation_index + 1]
            g_approx = (perturbated_output - initial_output) / step[perturbation_index]
            gradient.append(g_approx.real)

        return gradient

    def _compute_grad(
        self,
        input_values,  # type: ndarray
        n_perturbations,  # type: int
        input_perturbations,  # type: ndarray
        step,  # type: Union[float,ndarray]
        **kwargs  # type: Any
    ):  # type: (...) -> ndarray
        if step is None:
            step = self.step

        if not isinstance(step, ndarray):
            step = step * ones(n_perturbations)

        gradient = []
        initial_output = self.f_pointer(input_values, **kwargs)
        for perturbation_index in xrange(n_perturbations):
            perturbated_output = self.f_pointer(
                input_perturbations[:, perturbation_index], **kwargs
            )
            g_approx = (perturbated_output - initial_output) / step[perturbation_index]
            gradient.append(g_approx.real)

        return gradient

    def _get_opt_step(
        self,
        f_p,  # type: ndarray
        f_0,  # type: ndarray
        f_m,  # type: ndarray
        numerical_error=EPSILON,  # type: float
    ):  # type: (...) -> Tuple[ndarray,ndarray]
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
                By default Machine epsilon (appx 1e-16),
                but can be higher.
                when the calculation of :math:`f` requires a numerical resolution.

        Returns:
            * The errors.
            * The optimal steps.
        """
        n_out = f_p.size
        if n_out == 1:
            t_e, c_e, opt_step = comp_best_step(
                f_p, f_0, f_m, self.step, epsilon_mach=numerical_error
            )
            if t_e is None:
                error = 0.0
            else:
                error = t_e + c_e
        else:
            errors = zeros(n_out)
            opt_steps = zeros(n_out)
            for i in xrange(n_out):
                t_e, c_e, opt_steps[i] = comp_best_step(
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
        x_vect,  # type: ndarray
        numerical_error=EPSILON,  # type: float
        **kwargs
    ):  # type: (...) -> Tuple[ndarray,ndarray]
        r"""Compute the gradient by real step.

        Args:
            x_vect: The input vector.
            numerical_error: The numerical error
                associated to the calculation of :math:`f`.
                By default machine epsilon (appx 1e-16),
                but can be higher.
                when the calculation of :math:`f` requires a numerical resolution.
            **kwargs: The additional arguments passed to the function.

        Returns:
            * The optimal steps.
            * The errors.
        """
        n_dim = len(x_vect)
        x_p_arr, _ = self.generate_perturbations(n_dim, x_vect)
        x_m_arr, _ = self.generate_perturbations(n_dim, x_vect, step=-self.step)
        opt_steps = self.step * ones(n_dim)
        errors = zeros(n_dim)
        comp_step = self._get_opt_step
        if self._parallel:

            def func_noargs(
                xval,  # type: ndarray
            ):  # type: (...) -> ndarray
                """Call the function without explicitly passed arguments."""
                return self.f_pointer(xval, **kwargs)

            function_list = [func_noargs] * (n_dim + 1)
            parallel_execution = ParallelExecution(function_list, **self._par_args)

            all_x = [x_vect] + [x_p_arr[:, i] for i in xrange(n_dim)]
            all_x += [x_m_arr[:, i] for i in xrange(n_dim)]
            output_list = parallel_execution.execute(all_x)

            f_0 = output_list[0]
            for i in xrange(n_dim):
                f_p = output_list[i + 1]
                f_m = output_list[n_dim + i + 1]
                errs, opt_step = comp_step(
                    f_p, f_0, f_m, numerical_error=numerical_error
                )
                errors[i] = errs
                opt_steps[i] = opt_step
        else:
            f_0 = self.f_pointer(x_vect, **kwargs)
            for i in xrange(n_dim):
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
        input_values,  # type: ndarray
        input_indices,  # type: List[int]
        step,  # type: float
    ):  # type: (...) -> Tuple[ndarray,ndarray]
        input_dimension = len(input_values)
        n_indices = len(input_indices)
        input_perturbations = (
            tile(input_values, n_indices).reshape((n_indices, input_dimension)).T
        )
        if self._design_space is None:
            input_perturbations[input_indices, xrange(n_indices)] += step
            return input_perturbations, step
        else:
            if self._normalize:
                upper_bounds = self._design_space.normalize_vect(
                    self._design_space.get_upper_bounds()
                )
            else:
                upper_bounds = self._design_space.get_upper_bounds()

            steps = where(
                input_perturbations[input_indices, xrange(n_indices)] == upper_bounds,
                -step,
                step,
            )
            input_perturbations[input_indices, xrange(n_indices)] += steps

            return input_perturbations, steps


def comp_best_step(
    f_p,  # type: ndarray
    f_x,  # type: ndarray
    f_m,  # type: ndarray
    step,  # type: float
    epsilon_mach=EPSILON,  # type: float
):  # type: (...) -> Tuple[Optional[ndarray],Optional[ndarray],float]
    r"""Compute the optimal step for finite differentiation.

    Applied to a forward first order finite differences gradient approximation.

    Require a first evaluation of the perturbed functions values.

    The optimal step is reached when the truncation error
    (cut in the Taylor development),
    and the numerical cancellation errors
    (round-off when doing :math:`f(x+step)-f(x))` are equal.

    See:
    - https://en.wikipedia.org/wiki/Numerical_differentiation
    - *Numerical Algorithms and Digital Representation*,
      Knut Morken, Chapter 11, "Numerical Differenciation"

    Args:
        f_p: The value of the function :math:`f` at the next step :math:`x+\\delta_x`.
        f_x: The value of the function :math:`f` at the current step :math:`x`.
        f_m: The value of the function :math:`f` at the previous step :math:`x-\\delta_x`.
        step: The differentiation step :math:`\\delta_x`.

    Returns:
        * The estimation of the truncation error.
          None if the Hessian approximation is too small to compute the optimal step.
        * The estimation of the cancellation error.
          None if the Hessian approximation is too small to compute the optimal step.
        * The optimal step.
    """
    hess = approx_hess(f_p, f_x, f_m, step)

    if abs(hess) < 1e-10:
        LOGGER.debug("Hessian approximation is too small, can't compute optimal step.")
        return None, None, step

    opt_step = 2 * (epsilon_mach * abs(f_x) / abs(hess)) ** 0.5
    trunc_error = compute_truncature_error(hess, step)
    cancel_error = compute_cancellation_error(f_x, opt_step)
    return trunc_error, cancel_error, opt_step


def compute_truncature_error(
    hess,  # type: ndarray
    step,  # type: float
):  # type: (...) -> ndarray
    r"""Estimate the truncation error.

    Defined for a first order finite differences scheme.

    Args:
        hess: The second-order derivative :math:`d^2f/dx^2`.
        step: The differentiation step.

    Returns:
        The truncation error.
    """
    trunc_error = abs(hess) * step / 2
    return trunc_error


def compute_cancellation_error(
    f_x,  # type: ndarray
    step,  # type: float
    epsilon_mach=EPSILON,
):  # type: (...) -> ndarray
    r"""Estimate the cancellation error.

    This is the round-off when doing :math:`f(x+\\delta_x)-f(x)`.

    Args:
        f_x: The value of the function at the current step :math:`x`.
        step: The step used for the calculations of the perturbed functions values.
        epsilon_mach: The machine epsilon.

    Returns:
        The cancellation error.
    """
    epsa = epsilon_mach * abs(f_x)
    cancel_error = 2 * epsa / step
    return cancel_error


def approx_hess(
    f_p,  # type:ndarray
    f_x,  # type:ndarray
    f_m,  # type:ndarray
    step,  # type: float
):  # type: (...) -> ndarray
    r"""Compute the second-order approximation of the Hessian matrix :math:`d^2f/dx^2`.

    Args:
        f_p: The value of the function :math:`f` at the next step :math:`x+\\delta_x`.
        f_x: The value of the function :math:`f` at the current step :math:`x`.
        f_m: The value of the function :math:`f` at the previous step :math:`x-\\delta_x`.
        step: The differentiation step :math:`\\delta_x`.

    Returns:
        The approximation of the Hessian matrix at the current step :math:`x`.
    """
    hess = (f_p - 2 * f_x + f_m) / (step ** 2)
    return hess
