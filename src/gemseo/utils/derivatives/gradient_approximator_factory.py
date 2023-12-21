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
"""Factory for classes derived from :class:`GradientApproximator`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from gemseo.core.base_factory import BaseFactory
from gemseo.utils.derivatives.gradient_approximator import GradientApproximator

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.design_space import DesignSpace
    from gemseo.utils.derivatives.approximation_modes import ApproximationMode


class GradientApproximatorFactory(BaseFactory):
    """A factory to create gradient approximators.

    In addition to the names of the classes, the factory can be queried with an
    :class:`ApproximationMode`.
    """

    _CLASS = GradientApproximator
    _MODULE_NAMES = ("gemseo.utils.derivatives",)

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        for class_info in tuple(self._names_to_class_info.values()):
            approximation_mode = class_info.class_._APPROXIMATION_MODE
            self._names_to_class_info[approximation_mode] = class_info

    def create(
        self,
        name: str | ApproximationMode,
        f_pointer: Callable,
        step: float | complex | ndarray | None = None,
        design_space: DesignSpace | None = None,
        normalize: bool = True,
        parallel: bool = False,
        **parallel_args: Any,
    ) -> GradientApproximator:
        """Create a gradient approximator.

        Args:
            name: The name of the class or the approximation mode.
            f_pointer: The pointer to the function to derive.
            step: The default differentiation step.
            design_space: The design space
                containing the upper bounds of the input variables.
                If ``None``, consider that the input variables are unbounded.
            normalize: Whether to normalize the function.
            parallel: Whether to differentiate the function in parallel.
            **parallel_args: The parallel execution options,
                see :mod:`gemseo.core.parallel_execution`.

        Returns:
            The gradient approximator.
        """
        return super().create(
            name,
            f_pointer=f_pointer,
            step=step,
            design_space=design_space,
            normalize=normalize,
            parallel=parallel,
            **parallel_args,
        )

    @property
    def gradient_approximators(self) -> list[str]:
        """The gradient approximators."""
        return self.class_names
