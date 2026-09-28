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

"""A function that a problem wraps one of its own in."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.core.function.array_function import ArrayFunction
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.util.typing import NumberArray


class BaseProblemFunction(ArrayFunction, metaclass=ABCGoogleDocstringInheritanceMeta):
    """A function wrapping another one on behalf of a problem.

    A problem wraps each of its functions in two halves that stack.
    [EvaluationFunction][gemseo.core.function.evaluation_function.EvaluationFunction]
    evaluates the function,
    in the coordinates the user declared,
    and records what it evaluates;
    [TransformedInputFunction][gemseo.core.function.transformed_input_function.TransformedInputFunction]
    adapts the argument to the working coordinates,
    and wraps the first.
    A driver installs them with
    [bind_functions][gemseo.core.problem.evaluation.EvaluationProblem.bind_functions]
    and
    [create_working_problem][gemseo.core.problem.evaluation.EvaluationProblem.create_working_problem],
    in that order.

    Both halves answer to the same three things:
    whether an evaluation stops on a `NaN`,
    whether the calls are counted,
    and how many there have been.
    One half owns each of them and the other reads through to it,
    which is what lets a caller holding a function of a problem
    read them without knowing which half it holds.
    """  # noqa: E501

    _wrapped_function: ArrayFunction
    """The function this one wraps."""

    _gradient_name: str
    """The name under which a Jacobian is recorded in a database."""

    _vectorize: bool
    """Whether the function is evaluated on a matrix of samples.

    A vectorized evaluation asks the function for a matrix of samples at once
    and gets the block diagonal matrix of their Jacobians back.
    Both halves cut the same blocks out of that matrix,
    the adaptation one reading this setting from the evaluation one it wraps.
    """

    def __init__(
        self,
        function: ArrayFunction,
        with_normalized_inputs: bool = False,
    ) -> None:
        """
        Args:
            function: The function this one wraps.
            with_normalized_inputs: Whether this function takes a normalized point.
                The adaptation half passes `False`,
                since a point of the working space,
                and not a normalized one,
                is what it takes.
        """  # noqa: D205, D212
        self._wrapped_function = function
        super().__init__(
            self._compute_output,
            function.name,
            jac=self._compute_jacobian,
            f_type=function.f_type,
            expr=function.expr,
            input_names=function.input_names,
            dim=function.dim,
            output_names=function.output_names,
            force_real=function.force_real,
            special_repr=function.special_repr,
            original_name=function.original_name,
            with_normalized_inputs=with_normalized_inputs,
        )
        # `original` is set by each half after this call,
        # since `ArrayFunction.__init__` sets it to `self`;
        # it is transitive across the halves of one problem only,
        # and stops at a wrapper another problem installed.

    @abstractmethod
    def _compute_output(self, input_value: NumberArray) -> NumberArray:
        """Compute an output value.

        Args:
            input_value: The input value.

        Returns:
            The output value.
        """

    @abstractmethod
    def _compute_jacobian(self, input_value: NumberArray) -> NumberArray:
        """Compute a Jacobian.

        Args:
            input_value: The input value.

        Returns:
            The Jacobian.
        """

    @property
    @abstractmethod
    def stop_if_nan(self) -> bool:
        """Whether the evaluation stops when the function returns a `NaN`."""

    @stop_if_nan.setter
    @abstractmethod
    def stop_if_nan(self, value: bool) -> None: ...

    @property
    @abstractmethod
    def enable_statistics(self) -> bool:
        """Whether the calls of the function are counted."""

    @property
    @abstractmethod
    def n_calls(self) -> int:
        """The number of times the function has been evaluated."""

    @n_calls.setter
    @abstractmethod
    def n_calls(self, value: int) -> None: ...
