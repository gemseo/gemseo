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
#       :author: Damien Guenot - 26 avr. 2016
#       :author: Francois Gallard, refactoring
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class for algorithm libraries to handle a :class:`.BaseProblem`."""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from docstring_inheritance import GoogleDocstringInheritanceMeta

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.base_algorithm_settings import BaseAlgorithmSettings
from gemseo.algos.opt.base_gradient_based_algorithm_settings import (
    BaseGradientBasedAlgorithmSettings,
)
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.pydantic import create_model
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.base_problem import BaseProblem
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


@dataclass
class AlgorithmDescription(metaclass=GoogleDocstringInheritanceMeta):
    """The description of an algorithm."""

    algorithm_name: str
    """The name of the algorithm in |g|."""

    internal_algorithm_name: str
    """The name of the algorithm in the wrapped library."""

    library_name: str = ""
    """The name of the wrapped library."""

    description: str = ""
    """A description of the algorithm."""

    website: str = ""
    """The website of the wrapped library or algorithm."""

    # TODO: The field below is a workaround to be able to access algo-specific settings
    #  in modules for which one library class handles many different algorithms. In the
    #  future we should have one algorithm per module and use the Settings class
    #  variable to validate the settings in _validate_settings.
    Settings: type[BaseAlgorithmSettings] = BaseAlgorithmSettings
    """The Pydantic model for the settings."""


class BaseAlgorithmLibrary(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for algorithm libraries to handle a :class:`.BaseProblem`.

    An algorithm library solves a numerical problem (optim, doe, linear problem) using a
    particular algorithm from a particular family of numerical methods.

    Provide the available methods in the library for the proposed problem to be solved.

    To integrate an optimization package, inherit from this class and put your module in
    gemseo.algos.doe or gemseo.algo.opt, or gemseo.algos.linear_solver packages.
    """

    _algo_name: str
    """The name of the algorithm."""

    _problem: BaseProblem | None
    """The problem to be solved."""

    ALGORITHM_INFOS: ClassVar[dict[str, AlgorithmDescription]] = {}
    """The description of the algorithms contained in the library."""

    def __init__(self, algo_name: str) -> None:
        """
        Args:
            algo_name: The algorithm name.

        Raises:
            KeyError: When the algorithm is not in the library.
        """  # noqa: D205, D212
        if algo_name not in self.ALGORITHM_INFOS:
            msg = (
                f"The algorithm {algo_name} is unknown in {self.__class__.__name__}; "
                f"available ones are: {pretty_str(self.ALGORITHM_INFOS.keys())}."
            )
            raise KeyError(msg)

        self._algo_name = algo_name
        self._problem = None

    @property
    def algo_name(self) -> str:
        """The name of the algorithm."""
        return self._algo_name

    def _validate_settings(
        self,
        settings_model: BaseAlgorithmSettings | None = None,
        **settings: Any,
    ) -> dict[str, Any]:
        """Validate the settings with the appropriate Pydantic model.

        Args:
            settings_model: The algorithm settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **settings: The algorithm settings.
                These arguments are ignored when ``settings_model`` is not ``None``.

        Returns:
            The validated settings.
        """
        return create_model(
            self.ALGORITHM_INFOS[self.algo_name].Settings,
            settings_model=settings_model,
            **settings,
        ).model_dump()

    @staticmethod
    def _filter_settings(
        settings: StrKeyMapping,
        model_to_exclude: type[BaseAlgorithmSettings],
    ) -> dict[str, Any]:
        """Filter settings.

        Args:
            settings: The settings to be filtered.
            model_to_exclude: The model whose fields should be excluded from settings.

        Returns:
            The validated settings.
        """
        fields_to_exclude = model_to_exclude.model_fields

        # Remove GEMSEO settings lying in dedicated Pydantic models
        fields_to_exclude |= BaseGradientBasedAlgorithmSettings.model_fields

        return {key: settings[key] for key in settings if key not in fields_to_exclude}

    def _pre_run(
        self,
        problem: BaseProblem,
        **settings: Any,
    ) -> None:
        """Save the solver settings and name in the problem attributes.

        Args:
            problem: The problem to be solved.
            **settings: The settings for the algorithm, see associated JSON file.
        """

    def _post_run(
        self,
        problem: BaseProblem,
        result: ndarray,
        **settings: Any,
    ) -> None:
        """Save the LinearProblem to the disk when required.

        If the save_when_fail option is True, save the LinearProblem to the disk when
        the system failed and print the file name in the warnings.

        Args:
            problem: The problem to be solved.
            result: The result of the run, i.e. the solution.
            **settings: The settings for the algorithm.
        """

    def execute(
        self,
        problem: BaseProblem,
        settings_model: BaseAlgorithmSettings | None = None,
        **settings: Any,
    ) -> Any:
        """Solve a problem with an algorithm from this library.

        Args:
            problem: The problem to be solved.
            settings_model: The algorithm settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **settings: The algorithm settings.
                These arguments are ignored when ``settings_model`` is not ``None``.

        Returns:
            The solution found by the algorithm.
        """
        self._problem = problem
        problem.check()
        settings = self._validate_settings(settings_model=settings_model, **settings)
        self._pre_run(problem, **settings)
        self._run(problem, **settings)
        result = self._get_result(problem)
        self._post_run(problem, result, **settings)

        # Clear the state of _problem; the cache of the AlgoFactory can be used.
        self._problem = None
        return result

    @abstractmethod
    def _run(self, problem: BaseProblem, **settings: Any) -> None:
        """Solve the problem.

        Args:
            problem: The problem.
            **settings: The settings of the algorithm.
        """

    def _get_result(self, problem: BaseProblem) -> Any:
        """Return the result of the resolution of the problem.

        Args:
            problem: The problem.
        """

    def _check_algorithm(self, problem: BaseProblem) -> None:
        """Check that algorithm is available and adapted to the problem.

        Set the optimization library and the algorithm name according
        to the requirements of the optimization library.

        Args:
            problem: The problem to be solved.
        """
        algo_name = self._algo_name
        unsuitability_reason = self._get_unsuitability_reason(
            self.ALGORITHM_INFOS[algo_name], problem
        )
        if unsuitability_reason:
            msg = (
                f"The algorithm {algo_name} is not adapted to the problem "
                f"because {unsuitability_reason}."
            )
            raise ValueError(msg)

    @classmethod
    def _get_unsuitability_reason(
        cls, algorithm_description: AlgorithmDescription, problem: BaseProblem
    ) -> _UnsuitabilityReason:
        """Get the reason why an algorithm is not adapted to a problem.

        Args:
            algorithm_description: The description of the algorithm.
            problem: The problem to be solved.

        Returns:
            The reason why the algorithm is not adapted to the problem.
        """
        return _UnsuitabilityReason.NO_REASON

    @classmethod
    def is_algorithm_suited(
        cls, algorithm_description: AlgorithmDescription, problem: BaseProblem
    ) -> bool:
        """Check if an algorithm is suited to a problem according to its description.

        Args:
            algorithm_description: The description of the algorithm.
            problem: The problem to be solved.

        Returns:
            Whether the algorithm is suited to the problem.
        """
        return not cls._get_unsuitability_reason(algorithm_description, problem)

    @classmethod
    def filter_adapted_algorithms(cls, problem: BaseProblem) -> list[str]:
        """Filter the algorithms capable of solving the problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The names of the algorithms adapted to this problem.
        """
        adapted_algorithms = []
        for algo_name, algo_description in cls.ALGORITHM_INFOS.items():
            if cls.is_algorithm_suited(algo_description, problem):
                adapted_algorithms.append(algo_name)

        return adapted_algorithms
