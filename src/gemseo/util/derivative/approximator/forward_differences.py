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

from numpy import full
from numpy import ndarray
from numpy import tile
from numpy import where

from gemseo.util.derivative.approximation_mode import ApproximationMode
from gemseo.util.derivative.approximator.base_finite_differences import (
    BaseFiniteDifferences,
)

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray


class ForwardDifferences(BaseFiniteDifferences):
    r"""Finite differences approximator.

    $$\frac{df(x)}{dx}\approx\frac{f(x+\delta x)-f(x)}{\delta x}$$
    """

    _APPROXIMATION_MODE = ApproximationMode.FINITE_DIFFERENCES

    def _compute_parallel_grad(
        self,
        input_values: RealArray,
        input_perturbations: RealArray,
        step: float | RealArray,
        **kwargs: Any,
    ) -> list[RealArray]:
        n_perturbations = input_perturbations.shape[1]
        if not isinstance(step, ndarray):
            step = full(n_perturbations, step)

        self._function_kwargs = kwargs
        parallel_execution = self._create_callable_parallel_execution(
            self._wrap_function,
            self._parallel_args.get("use_threading", False),
            n_perturbations + 1,
        )

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
        input_values: RealArray,
        input_perturbations: RealArray,
        step: float | RealArray,
        **kwargs: Any,
    ) -> list[RealArray]:
        n_perturbations = input_perturbations.shape[1]
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

    def _generate_perturbations(
        self,
        input_values: RealArray,
        input_indices: list[int],
        step: float,
    ) -> tuple[RealArray, float | RealArray]:
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
