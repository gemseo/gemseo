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

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import atleast_1d
from openturns import CompositeRandomVector
from openturns import Greater
from openturns import GreaterOrEqual
from openturns import IntersectionEvent
from openturns import Less
from openturns import LessOrEqual
from openturns import PythonFunction
from openturns import RandomGenerator
from openturns import RandomVector
from openturns import ThresholdEvent
from openturns import UnionEvent

from gemseo.uncertainty.reliability.core.base import BaseReliabilityAlgorithm
from gemseo.uncertainty.reliability.threshold_comparator import ThresholdComparator

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping

    from openturns import PersistentObject

    from gemseo.core.function.array_function import OutputType
    from gemseo.uncertainty.reliability.elementary_event import ElementaryEvent
    from gemseo.uncertainty.reliability.problem import ReliabilityProblem
    from gemseo.util.typing import NumberArray
    from gemseo.util.typing import RealArray


_comparator_to_ot_comparator: Final[
    Mapping[ThresholdComparator, type[PersistentObject]]
] = MappingProxyType({
    ThresholdComparator.LESS: Less,
    ThresholdComparator.LESS_EQUAL: LessOrEqual,
    ThresholdComparator.GREATER: Greater,
    ThresholdComparator.GREATER_EQUAL: GreaterOrEqual,
})
"""The map from a comparator to its OpenTURNS comparator."""


def _create_intersection_event(
    events: Iterable[ThresholdEvent | UnionEvent],
) -> IntersectionEvent:
    """Combine OpenTURNS events with an OpenTURNS intersection.

    Args:
        events: The OpenTURNS events of a single intersection.

    Returns:
        Their OpenTURNS intersection.
    """
    return IntersectionEvent(list(events))


def _create_union_event(events: Iterable[IntersectionEvent]) -> UnionEvent:
    """Combine OpenTURNS events with an OpenTURNS union.

    Args:
        events: The OpenTURNS events of the intersections.

    Returns:
        Their OpenTURNS union.
    """
    return UnionEvent(list(events))


class BaseOTReliabilityAlgorithm(BaseReliabilityAlgorithm):
    """The base class for the OpenTURNS-based reliability analysis algorithms."""

    _algo_class: ClassVar[type[PersistentObject]]
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
        random_space = problem.input_space
        input_vector = RandomVector(random_space.variables.distribution.distribution)
        dimension = random_space.dimension
        observables = {function.name: function for function in problem.observables}
        event = problem.name_to_event[event_name]

        def create_threshold_event(
            elementary_event: ElementaryEvent,
        ) -> ThresholdEvent:
            """Create the OpenTURNS threshold event of an elementary event.

            Args:
                elementary_event: The elementary event.

            Returns:
                The OpenTURNS threshold event.
            """
            # Use the evaluation function related to event.function
            function = observables[elementary_event.function.name]
            func = _FunctionForOpenTURNS(function.evaluate, False)
            jac = (
                _FunctionForOpenTURNS(function.jac, True)
                if elementary_event.function.has_jac
                else None
            )
            ot_function = PythonFunction(dimension, 1, func, gradient=jac)
            output_vector = CompositeRandomVector(ot_function, input_vector)
            comparator = _comparator_to_ot_comparator[elementary_event.comparator]()
            return ThresholdEvent(output_vector, comparator, elementary_event.threshold)

        if not event.is_combination:
            # A single elementary event stays a bare ThresholdEvent,
            # as OpenTURNS algorithms expect it unwrapped in this case.
            (elementary_event,) = next(iter(event))
            return create_threshold_event(elementary_event)

        return event._fold(
            create_threshold_event, _create_intersection_event, _create_union_event
        )

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
        """  # noqa: D205, D212
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
