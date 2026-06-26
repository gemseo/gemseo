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
"""Base class for the OpenTURNS-based reliability analysis algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from numpy import atleast_1d
from openturns import CompositeRandomVector
from openturns import Greater
from openturns import IntersectionEvent
from openturns import Less
from openturns import PythonFunction
from openturns import RandomGenerator
from openturns import RandomVector
from openturns import ThresholdEvent
from openturns import UnionEvent

from gemseo.uncertainty.reliability.base import BaseReliabilityAlgorithm

if TYPE_CHECKING:
    from collections.abc import Callable

    from openturns import PersistentObject

    from gemseo.core.functions.array_function import OutputType
    from gemseo.typing import NumberArray
    from gemseo.typing import RealArray
    from gemseo.uncertainty.reliability.problem import ReliabilityProblem


class BaseOTReliabilityAlgorithm(BaseReliabilityAlgorithm):
    """The base class for the OpenTURNS-based reliability analysis algorithms."""

    _ALGO_CLASS: ClassVar[type[PersistentObject]]
    """The OpenTURNS class to instantiate the reliability analysis algorithm."""

    @staticmethod
    def _create_ot_event(
        event_name: str,
        problem: ReliabilityProblem,
    ) -> ThresholdEvent | UnionEvent:
        """Create the OpenTURNS event related to an event.

        Args:
            event_name: The name of the event.
            problem: The reliability analysis problem.

        Returns:
            The OpenTURNS event.
        """
        uncertain_space = problem.design_space
        input_vector = RandomVector(uncertain_space.distribution.distribution)
        dimension = uncertain_space.dimension
        observables = {function.name: function for function in problem.observables}
        ot_intersection_events = []
        event = problem.name_to_event[event_name]
        for intersection_event in event:
            ot_intersection_event = []
            ot_intersection_events.append(ot_intersection_event)
            for elementary_event in intersection_event:
                # Use the ProblemFunction related to event.function
                function = observables[elementary_event.function.name]
                func = _FunctionForOpenTURNS(function.evaluate, False)
                jac = (
                    _FunctionForOpenTURNS(function.jac, True)
                    if elementary_event.function.has_jac
                    else None
                )
                ot_function = PythonFunction(dimension, 1, func, gradient=jac)
                output_vector = CompositeRandomVector(ot_function, input_vector)
                comparator = Greater() if elementary_event.greater else Less()
                ot_elementary_event = ThresholdEvent(
                    output_vector, comparator, elementary_event.threshold
                )
                ot_intersection_event.append(ot_elementary_event)

        if not event.is_combination:
            return ot_intersection_events[0][0]

        return UnionEvent([
            IntersectionEvent(ot_intersection_event)
            for ot_intersection_event in ot_intersection_events
        ])

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Set the seed for reliability analysis algorithm.

        Args:
            seed: The seed for reliability analysis algorithm.
        """
        RandomGenerator.SetSeed(seed)


class _FunctionForOpenTURNS:
    """`ArrayFunction` wrapper to be used by `openturns.PythonFunction`."""

    __function: Callable[[NumberArray], OutputType]
    """The wrapped function."""

    __is_jacobian: bool
    """Whether the function is a Jacobian function."""

    def __init__(
        self, function: Callable[[NumberArray], OutputType], is_jacobian: bool
    ) -> None:
        """
        Args:
            function: The function to be wrapped.
            is_jacobian: Whether the function is a Jacobian function.
        """  # noqa: D205 D212
        self.__function = function
        self.__is_jacobian = is_jacobian

    def __call__(self, input_value) -> RealArray:
        """Evaluate the function.

        Args:
            input_value: The input value of the function.

        Returns:
            The output value of the function.
        """
        result = atleast_1d(self.__function(array(input_value)))
        # openturns.PythonFunction expects an output value shaped as (d,)
        # and a Jacobian value shaped as (d, 1).
        return result.reshape((result.size, 1)) if self.__is_jacobian else result
