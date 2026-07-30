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

from numpy import concatenate
from numpy import tile
from numpy import where
from numpy.linalg import norm

from gemseo.util.derivative.approximation_mode import ApproximationMode
from gemseo.util.derivative.approximator.base_finite_differences import (
    BaseFiniteDifferences,
)

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class CenteredDifferences(BaseFiniteDifferences):
    r"""Centered differences approximator.

    $$\frac{df(x)}{dx}\approx\frac{f(x+\delta x)-f(x-\delta x)}{2\delta x}$$
    """

    _APPROXIMATION_MODE = ApproximationMode.CENTERED_DIFFERENCES

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
        parallel_execution = self._create_callable_parallel_execution(
            self._wrap_function,
            self._parallel_args.get("use_threading", False),
            n_perturbations,
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
                strict=False,
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
                strict=False,
            )
        ]

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
