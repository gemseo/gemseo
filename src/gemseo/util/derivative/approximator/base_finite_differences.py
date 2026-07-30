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

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import argmax
from numpy import atleast_1d
from numpy import full
from numpy import zeros

from gemseo.util.derivative.approximator.base import BaseGradientApproximator
from gemseo.util.derivative.error_estimator import EPSILON
from gemseo.util.derivative.error_estimator import compute_best_step

if TYPE_CHECKING:
    from numpy import floating

    from gemseo.util.typing import RealArray


class BaseFiniteDifferences(BaseGradientApproximator):
    """A base class for gradient approximation by finite differences."""

    _DEFAULT_STEP: ClassVar[float] = 1.0e-6

    def _get_opt_step(
        self,
        f_p: RealArray,
        f_0: RealArray,
        f_m: RealArray,
        numerical_error: float = EPSILON,
    ) -> tuple[floating, floating]:
        r"""Compute the optimal step of a function.

        This function may be a vector function.
        In this case, take the worst case.

        Args:
            f_p: The value of the function $f$
                 at the next step $x+\delta_x$.
            f_0: The value of the function $f$
                 at the current step $x$.
            f_m: The value of the function $f$
                 at the previous step $x-\delta_x$.
            numerical_error: The numerical error
                associated to the calculation of $f$.
                By default, Machine epsilon (appx 1e-16),
                but can be higher.
                when the calculation of $f$ requires a numerical resolution.

        Returns:
            The errors and the optimal steps.
        """
        # The function may return a scalar, a 0-dimensional array or an array;
        # cast to 1D so that the outputs can be iterated over component-wise.
        f_p = atleast_1d(f_p)
        f_0 = atleast_1d(f_0)
        f_m = atleast_1d(f_m)
        n_out = f_p.size
        errors = zeros(n_out)
        opt_steps = zeros(n_out)
        for i in range(n_out):
            t_e, c_e, opt_step = compute_best_step(
                f_p[i], f_0[i], f_m[i], self.step, epsilon_mach=numerical_error
            )
            # compute_best_step returns scalars or arrays depending on whether
            # ``self.step`` is a scalar or an array; cast to 1D before indexing.
            opt_steps[i] = atleast_1d(opt_step)[0]
            if t_e is not None:
                errors[i] = atleast_1d(t_e)[0] + atleast_1d(c_e)[0]

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
                associated to the calculation of $f$.
                By default, machine epsilon (appx 1e-16),
                but can be higher.
                when the calculation of $f$ requires a numerical resolution.
            **kwargs: The additional arguments passed to the function.

        Returns:
            The optimal steps and the errors.
        """
        n_dim = len(x_vect)
        x_p_arr, _ = self.generate_perturbations(n_dim, x_vect)
        x_m_arr, _ = self.generate_perturbations(n_dim, x_vect, step=-self.step)
        opt_steps = full(n_dim, self.step)
        errors = zeros(n_dim)
        comp_step = self._get_opt_step
        if self._parallel:
            self._function_kwargs = kwargs
            parallel_execution = self._create_callable_parallel_execution(
                self._wrap_function,
                self._parallel_args.get("use_threading", False),
                n_dim * 2 + 1,
            )
            all_x = [x_vect] + [x_p_arr[:, i] for i in range(n_dim)]
            all_x += [x_m_arr[:, i] for i in range(n_dim)]
            outputs = parallel_execution.execute(all_x)

            f_0 = outputs[0]
            for i in range(n_dim):
                f_p = outputs[i + 1]
                f_m = outputs[n_dim + i + 1]
                errors[i], opt_steps[i] = comp_step(
                    f_p, f_0, f_m, numerical_error=numerical_error
                )
        else:
            f_0 = self.f_pointer(x_vect, **kwargs)
            for i in range(n_dim):
                f_p = self.f_pointer(x_p_arr[:, i], **kwargs)
                f_m = self.f_pointer(x_m_arr[:, i], **kwargs)
                errors[i], opt_steps[i] = comp_step(
                    f_p, f_0, f_m, numerical_error=numerical_error
                )
        self.step = opt_steps
        return opt_steps, errors
