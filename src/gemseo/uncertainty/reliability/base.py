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
"""Base class for the reliability analysis algorithms."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from gemseo.uncertainty.reliability.base_settings import (
        BaseReliabilityAlgorithmSettings,
    )
    from gemseo.uncertainty.reliability.problem import ReliabilityProblem
    from gemseo.uncertainty.reliability.result import ReliabilityResult


class BaseReliabilityAlgorithm(metaclass=ABCGoogleDocstringInheritanceMeta):
    """The base class for the reliability analysis algorithms.

    A reliability analysis algorithm estimates
    probabilities of events from a reliability problem.
    In addition to elementary events,
    e.g. $y > 3$,
    some algorithms support combinations of elementary events,
    e.g. $(y > 3) & (z < 8)$.
    """

    # TODO: subclass BaseAlgo once the MR 2434 has been merged.

    settings_class: ClassVar[type[BaseReliabilityAlgorithmSettings]]
    """The type of settings for the reliability analysis algorithm."""

    SUPPORT_ELEMENTARY_EVENT_COMBINATIONS: ClassVar[bool] = False
    """Whether the reliability analysis algorithm supports combinations of elementary events."""  # noqa: E501

    def execute(
        self,
        problem: ReliabilityProblem,
        settings: BaseReliabilityAlgorithmSettings | None = None,
    ) -> dict[str, ReliabilityResult]:
        """Estimate probabilities of events.

        Args:
            problem: The reliability analysis problem defining the events.
            settings: The settings of the reliability analysis algorithm.
                If `None`, use the default settings.

        Returns:
            The map from an event name to a reliability analysis result.

        Raises:
            ValueError: When the reliability analysis problem has no event.
            TypeError: When the reliability analysis algorithm does not support
                combinations of elementary events
                but the reliability analysis problem contains such combinations.
        """
        algo_name = self.__class__.__name__
        name_to_event = problem.name_to_event
        if not name_to_event:
            msg = (
                f"{algo_name} requires a reliability analysis problem "
                "with at least one event."
            )
            raise ValueError(msg)

        for event in name_to_event.values():
            if event.is_combination and not self.SUPPORT_ELEMENTARY_EVENT_COMBINATIONS:
                msg = f"{algo_name} does not support combinations of elementary events."
                raise TypeError(msg)

        if settings is None:
            settings = self.settings_class()

        problem.preprocess_functions(
            is_function_input_normalized=False, use_database=settings.use_database
        )

        results = {}
        for event_name in name_to_event:
            results[event_name] = self._execute(event_name, problem, settings)

        return results

    @abstractmethod
    def _execute(
        self,
        event_name: str,
        problem: ReliabilityProblem,
        settings: BaseReliabilityAlgorithmSettings,
    ) -> ReliabilityResult:
        """Estimate the probability of an event.

        Args:
            event_name: The name of the event.
            problem: The reliability analysis problem.
            settings: The settings of the reliability analysis algorithm.

        Returns:
            The reliability analysis result.
        """
