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
"""Gradient approximation."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar

from numpy import array
from numpy import float64
from numpy import ndarray

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Number

    from gemseo.algos.design_space import DesignSpace
    from gemseo.utils.derivatives.approximation_modes import ApproximationMode


class GradientApproximator(metaclass=ABCGoogleDocstringInheritanceMeta):
    """A gradient approximator."""

    f_pointer: Callable[[ndarray], ndarray]
    """The pointer to the function to derive."""

    _APPROXIMATION_MODE: ClassVar[ApproximationMode]
    """The approximation mode that a derived class implements."""

    _DEFAULT_STEP: ClassVar[Number]
    """The default value for the step."""

    def __init__(
        self,
        f_pointer: Callable[[ndarray], ndarray],
        step: float | complex | ndarray | None = None,
        design_space: DesignSpace | None = None,
        normalize: bool = True,
        parallel: bool = False,
        **parallel_args: Any,
    ) -> None:
        """
        Args:
            f_pointer: The pointer to the function to derive.
            step: The default differentiation step.
            design_space: The design space
                containing the upper bounds of the input variables.
                If ``None``, consider that the input variables are unbounded.
            normalize: Whether to normalize the function.
            parallel: Whether to differentiate the function in parallel.
            **parallel_args: The parallel execution options,
                see :mod:`gemseo.core.parallel_execution`.
        """  # noqa:D205 D212 D415
        self.f_pointer = f_pointer
        self._parallel_args = parallel_args
        self._parallel = parallel
        self.step = step if step is not None else self._DEFAULT_STEP
        self._design_space = design_space
        self._normalize = normalize
        self._function_kwargs = {}

    @property
    def step(self) -> float:
        """The default approximation step."""
        return self._step

    @step.setter
    def step(
        self,
        value: float,
    ) -> None:
        self._step = value

    def f_gradient(
        self,
        x_vect: ndarray,
        step: float | None = None,
        x_indices: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> ndarray:
        """Approximate the gradient of the function for a given input vector.

        Args:
            x_vect: The input vector.
            step: The differentiation step.
                If ``None``, use the default differentiation step.
            x_indices: The components of the input vector
                to be used for the differentiation.
                If ``None``, use all the components.
            **kwargs: The optional arguments for the function.

        Returns:
            The approximated gradient.
        """
        input_dimension = len(x_vect)
        input_perturbations, steps = self.generate_perturbations(
            input_dimension, x_vect, x_indices=x_indices, step=step
        )
        n_perturbations = input_perturbations.shape[1]

        self._function_kwargs = kwargs

        if self._parallel:
            grad = self._compute_parallel_grad(
                x_vect, n_perturbations, input_perturbations, steps, **kwargs
            )
        else:
            grad = self._compute_grad(
                x_vect, n_perturbations, input_perturbations, steps, **kwargs
            )

        return array(grad, dtype=float64).T

    # TODO: API: remove useless n_perturbations
    @abstractmethod
    def _compute_parallel_grad(
        self,
        input_values: ndarray,
        n_perturbations: int,
        input_perturbations: ndarray,
        step: float,
        **kwargs: Any,
    ) -> ndarray:
        """Approximate the gradient in parallel.

        Args:
            input_values: The input values.
            n_perturbations: The number of perturbations.
            input_perturbations: The perturbations of the input.
            step: The differentiation step,
                either one global step or one step by input component.
            **kwargs: The optional arguments for the function.

        Returns:
            The approximated gradient.
        """

    @abstractmethod
    def _compute_grad(
        self,
        input_values: ndarray,
        n_perturbations: int,
        input_perturbations: ndarray,
        step: float | ndarray,
        **kwargs: Any,
    ) -> ndarray:
        """Approximate the gradient.

        Args:
            input_values: The input values.
            n_perturbations: The number of perturbations.
            input_perturbations: The input perturbations.
            step: The differentiation step,
                either one global step or one step by input component.
            **kwargs: The optional arguments for the function.

        Returns:
            The approximated gradient.
        """

    def generate_perturbations(
        self,
        n_dim: int,
        x_vect: ndarray,
        x_indices: Sequence[int] | None = None,
        step: float | None = None,
    ) -> tuple[ndarray, float | ndarray]:
        """Generate the input perturbations from the differentiation step.

        These perturbations will be used to compute the output ones.

        Args:
            n_dim: The input dimension.
            x_vect: The input vector.
            x_indices: The components of the input vector
                to be used for the differentiation.
                If ``None``, use all the components.
            step: The differentiation step.
                If ``None``, use the default differentiation step.

        Returns:
            * The input perturbations.
            * The differentiation step,
              either one global step or one step by input component.
        """
        if step is None:
            step = self.step

        if x_indices is None:
            x_indices = range(n_dim)

        return self._generate_perturbations(x_vect, x_indices, step)

    @abstractmethod
    def _generate_perturbations(
        self,
        input_values: ndarray,
        input_indices: list[int],
        step: float,
    ) -> tuple[ndarray, float | ndarray]:
        """Generate the input perturbations from the differentiation step.

        These perturbations will be used to compute the output ones.

        Args:
            input_values: The input vector.
            input_indices: The components of the input vector
                to be used for the differentiation.
            step: The differentiation step.

        Returns:
            * The input perturbations.
            * The differentiation step,
              either one global step or one step by input component.
        """

    def _wrap_function(
        self,
        f_input_values: ndarray,
    ) -> ndarray:
        """Wrap the function to be called without explicitly passed arguments.

        Args:
            f_input_values: The input values.

        Return:
            The value of the function output.
        """
        return self.f_pointer(f_input_values, **self._function_kwargs)
