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

"""Gradient approximation."""
from __future__ import division, unicode_literals

import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import six
from custom_inherit import DocInheritMeta
from numpy import array, finfo, float64, ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.core.factory import Factory
from gemseo.utils.py23_compat import xrange

EPSILON = finfo(float).eps
LOGGER = logging.getLogger(__name__)


@six.add_metaclass(
    DocInheritMeta(
        abstract_base_class=True,
        style="google_with_merge",
        include_special_methods=True,
    )
)
class GradientApproximator(object):
    """A gradient approximator.

    Attributes:
        f_pointer (Callable): The pointer to the function to derive.
    """

    ALIAS = None

    def __init__(
        self,
        f_pointer,  # type: Callable[[ndarray],ndarray]
        step=1e-6,  # type: float
        parallel=False,  # type: bool
        design_space=None,  # type: Optional[DesignSpace]
        normalize=True,  # type: bool
        **parallel_args  # type: Union[int,bool,float]
    ):  # type: (...) -> None
        """
        Args:
            f_pointer: The pointer to the function to derive.
            step: The default differentiation step.
            parallel: Whether to differentiate the function in parallel.
            design_space: The design space
                containing the upper bounds of the input variables.
                If None, consider that the input variables are unbounded.
            normalize: If True, then the functions are normalized.
            **parallel_args: The parallel execution options,
                see :mod:`gemseo.core.parallel_execution`.
        """
        self.f_pointer = f_pointer
        self.__par_args = parallel_args
        self.__parallel = parallel
        self._step = None
        self.step = step
        self._design_space = design_space
        self._normalize = normalize

    @property
    def _parallel(self):  # type: (...) -> bool
        """Whether to differentiate the function in parallel."""
        return self.__parallel

    @property
    def _par_args(self):  # type: (...) -> Union[int,bool,float]
        """The parallel execution options."""
        return self.__par_args

    @property
    def step(self):  # type: (...) -> float
        """The default approximation step."""
        return self._step

    @step.setter
    def step(
        self,
        value,  # type: float
    ):  # type: (...) -> None
        self._step = value

    def f_gradient(
        self,
        x_vect,  # type: ndarray
        step=None,  # type: Optional[float]
        x_indices=None,  # type: Optional[Sequence[int]]
        **kwargs  # type: Any
    ):  # type: (...) -> ndarray
        """Approximate the gradient of the function for a given input vector.

        Args:
            x_vect: The input vector.
            step: The differentiation step.
                If None, use the default differentiation step.
            x_indices: The components of the input vector
                to be used for the differentiation.
                If None, use all the components.
            **kwargs: The optional arguments for the function.

        Returns:
            The approximated gradient.
        """
        input_dimension = len(x_vect)
        input_perturbations, steps = self.generate_perturbations(
            input_dimension, x_vect, x_indices=x_indices, step=step
        )
        n_perturbations = input_perturbations.shape[1]

        if self._parallel:
            grad = self._compute_parallel_grad(
                x_vect, n_perturbations, input_perturbations, steps, **kwargs
            )
        else:
            grad = self._compute_grad(
                x_vect, n_perturbations, input_perturbations, steps, **kwargs
            )

        return array(grad, dtype=float64).T

    def _compute_parallel_grad(
        self,
        input_values,  # type: ndarray
        n_perturbations,  # type: int
        input_perturbations,  # type: ndarray
        step,  # type: float
        **kwargs  # type: Any
    ):  # type: (...) -> ndarray
        """Approximate the gradient in parallel.

        Args:
            input_values: The input values.
            n_perturbations: The number of perturbations.
            step: The differentiation step,
                either one global step or one step by input component.
            **kwargs: The optional arguments for the function.

        Returns:
            The approximated gradient.
        """
        raise NotImplementedError

    def _compute_grad(
        self,
        input_values,  # type: ndarray
        n_perturbations,  # type: int
        input_perturbations,  # type: ndarray
        step,  # type: Union[float,ndarray]
        **kwargs  # type: Any
    ):  # type: (...) -> ndarray
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
        raise NotImplementedError

    def generate_perturbations(
        self,
        n_dim,  # type: int
        x_vect,  # type: ndarray
        x_indices=None,  # type: Optional[Sequence[int]]
        step=None,  # type: Optional[float]
    ):  # type: (...) -> Tuple[ndarray,Union[float,ndarray]]
        """Generate the input perturbations from the differentiation step.

        These perturbations will be used to compute the output ones.

        Args:
            n_dim: The input dimension.
            x_vect: The input vector.
            x_indices: The components of the input vector
                to be used for the differentiation.
                If None, use all the components.
            step: The differentiation step.
                If None, use the default differentiation step.

        Returns:
            * The input perturbations.
            * The differentiation step,
              either one global step or one step by input component.
        """
        if step is None:
            step = self.step

        if x_indices is None:
            x_indices = xrange(n_dim)

        return self._generate_perturbations(x_vect, x_indices, step)

    def _generate_perturbations(
        self,
        input_values,  # type: ndarray
        input_indices,  # type: List[int]
        step,  # type: float
    ):  # type: (...) -> Tuple[ndarray,Union[float,ndarray]]
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
        raise NotImplementedError


class GradientApproximationFactory(object):
    """A factory to create gradient approximators."""

    def __init__(self):  # type: (...) -> None
        self.factory = Factory(GradientApproximator, ("gemseo.utils.derivatives",))
        self.__aliases = {
            self.factory.get_class(class_name).ALIAS: class_name
            for class_name in self.factory.classes
        }

    def create(
        self,
        name,  # type:str
        f_pointer,  # type: Callable
        step=None,  # type: Optional[float]
        parallel=False,  # type: bool
        **parallel_args
    ):  # type: (...) -> GradientApproximator
        """Create a gradient approximator.

        Args:
            name: Either the name of the class implementing the gradient approximator
                or its alias.
            f_pointer: The pointer to the function to differentiate.
            step: The differentiation step.
                If None, use a default differentiation step.
            parallel: Whether to differentiate the function in parallel.
            **parallel_args: The parallel execution options,
                see :mod:`gemseo.core.parallel_execution`.

        Returns:
            The gradient approximator.
        """
        if name in self.__aliases:
            name = self.__aliases[name]

        if step is None:
            return self.factory.create(
                name, f_pointer=f_pointer, parallel=parallel, **parallel_args
            )
        else:
            return self.factory.create(
                name, f_pointer=f_pointer, step=step, parallel=parallel, **parallel_args
            )

    @property
    def gradient_approximators(self):  # type: (...) -> List[str]
        """The gradient approximators."""
        return self.factory.classes

    def is_available(self, class_name):  # type:(...) -> bool
        """Whether a gradient approximator is available.

        Args:
            class_name: The name of the class implementing the gradient approximator.

        Return:
            Whether the gradient approximator is available.
        """
        return self.factory.is_available(class_name)
