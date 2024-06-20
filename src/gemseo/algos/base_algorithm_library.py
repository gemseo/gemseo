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

import inspect
import logging
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from docstring_inheritance import GoogleDocstringInheritanceMeta

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.source_parsing import get_options_doc
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.base_problem import BaseProblem
    from gemseo.typing import MutableStrKeyMapping
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

    _problem: Any | None
    """The problem to be solved."""

    _option_grammar: JSONGrammar | None
    """The grammar defining the options of the current algorithm."""

    _OPTIONS_DIR: ClassVar[str | Path] = "options"
    """The name of the directory containing the files of the grammars of the options."""

    _OPTIONS_MAP: ClassVar[dict[str, str]] = {}
    """The names of the options in |g| mapping to those in the wrapped library."""

    _COMMON_OPTIONS_GRAMMAR: ClassVar[JSONGrammar] = JSONGrammar("AlgoLibOptions")
    """The grammar defining the options common to all the algorithms of the library."""

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
        self._option_grammar = None
        self._init_options_grammar()

    @property
    def algo_name(self) -> str:
        """The name of the algorithm."""
        return self._algo_name

    def _init_options_grammar(self) -> JSONGrammar:
        """Initialize the options' grammar.

        Returns:
            The grammar of options.
        """
        algo_name = self._algo_name
        # Store the lib in case we rerun the same algorithm,
        # for multilevel scenarios for instance
        # This significantly speedups the process
        # because of the option grammar that is long to create
        if self._option_grammar is not None and self._option_grammar.name == algo_name:
            return self._option_grammar

        library_directory = Path(inspect.getfile(self.__class__)).parent
        options_directory = library_directory / self._OPTIONS_DIR
        algo_schema_file = options_directory / f"{algo_name.upper()}_options.json"
        lib_schema_file = (
            options_directory / f"{self.__class__.__name__.upper()}_options.json"
        )

        if algo_schema_file.exists():
            schema_file = algo_schema_file
        elif lib_schema_file.exists():
            schema_file = lib_schema_file
        else:
            msg = (
                f"Neither the options grammar file {algo_schema_file} for the "
                f"algorithm '{algo_name}' "
                f"nor the options grammar file {lib_schema_file} for the library "
                f"'{self.__class__.__name__}' has been found."
            )
            raise ValueError(msg)

        self._option_grammar = JSONGrammar(f"{algo_name}_algorithm_options")
        self._option_grammar.update(self._COMMON_OPTIONS_GRAMMAR)
        self._option_grammar.update_from_file(schema_file)
        self._option_grammar.set_descriptions(
            get_options_doc(self.__class__._get_options)
        )

        return self._option_grammar

    def _pre_run(
        self,
        problem: BaseProblem,
        **options: Any,
    ) -> None:
        """Save the solver options and name in the problem attributes.

        Args:
            problem: The problem to be solved.
            **options: The options for the algorithm, see associated JSON file.
        """

    def _post_run(
        self,
        problem: BaseProblem,
        result: ndarray,
        **options: Any,
    ) -> None:
        """Save the LinearProblem to the disk when required.

        If the save_when_fail option is True, save the LinearProblem to the disk when
        the system failed and print the file name in the warnings.

        Args:
            problem: The problem to be solved.
            result: The result of the run, i.e. the solution.
            **options: The options for the algorithm, see associated JSON file.
        """

    def _process_specific_option(
        self,
        options: MutableStrKeyMapping,
        option_key: str,
    ) -> None:
        """Preprocess the option specifically.

        Args:
            options: The options to be preprocessed.
            option_key: The current option key to process.
        """

    def _process_options(self, **options: Any) -> dict[str, Any]:
        """Convert the options to algorithm specific options and check them.

        Args:
            **options: The driver options.

        Returns:
            The converted options.

        Raises:
            ValueError: If an option is invalid.
        """
        for option_name in list(options.keys()):  # Copy keys on purpose
            # Remove extra options added in the _get_option method of the
            # driver
            if option_name not in self._option_grammar:
                del options[option_name]
            else:
                self._process_specific_option(options, option_name)

        self._option_grammar.validate(options)

        for option_name in list(options.keys()):  # Copy keys on purpose
            lib_option_name = self._OPTIONS_MAP.get(option_name)
            # Overload with specific keys
            if lib_option_name is not None:
                options[lib_option_name] = options[option_name]
                if lib_option_name != option_name:
                    del options[option_name]

        return options

    def _check_ignored_options(self, options: StrKeyMapping) -> None:
        """Check that the user did not pass options that do not exist for this driver.

        Log a warning if it is the case.

        Args:
            options: The options.
        """
        for option_name in options:
            if option_name not in self._option_grammar:
                msg = "Driver %s has no option %s, option is ignored."
                LOGGER.warning(msg, self._algo_name, option_name)

    def execute(
        self,
        problem: BaseProblem,
        **settings: Any,
    ) -> Any:
        """Solve a problem with an algorithm from this library.

        Args:
            problem: The problem to be solved.
            **settings: The algorithm settings.

        Returns:
            The solution found by the algorithm.
        """
        self._problem = problem
        settings = self._update_algorithm_options(**settings)
        problem.check()
        self._pre_run(problem, **settings)
        result = self._run(problem, **settings)
        self._post_run(problem, result, **settings)
        # Clear the state of _problem; the cache of the AlgoFactory can be used.
        self._problem = None
        return result

    def _update_algorithm_options(
        self, initialize_options_grammar: bool = True, **options: Any
    ) -> dict[str, Any]:
        """Update the algorithm options.

        1. Load the grammar of algorithm options.
        2. Warn about the ignored initial algorithm options.
        3. Complete the initial algorithm options with the default algorithm options.

        Args:
            initialize_options_grammar: Whether to initialize the grammar of options.
            **options: The initial algorithm options.

        Returns:
            The updated algorithm options.
        """
        if initialize_options_grammar:
            self._init_options_grammar()
        self._check_ignored_options(options)
        return self._get_options(**options)

    @abstractmethod
    def _get_options(self, **options: Any) -> dict[str, Any]:
        """Retrieve the options of the library.

        To be overloaded by subclasses.
        Used to define default values for options using keyword arguments.

        Args:
            **options: The options of the algorithm.

        Returns:
            The options of the algorithm.
        """

    @abstractmethod
    def _run(self, problem: BaseProblem, **options: Any) -> Any:
        """Run the algorithm.

        Args:
            problem: The problem to be solved.
            **options: The options of the algorithm.

        Returns:
            The solution of the problem.
        """

    def _check_algorithm(self, problem: Any) -> None:
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
        cls, algorithm_description: AlgorithmDescription, problem: Any
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
        cls, algorithm_description: AlgorithmDescription, problem: Any
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
    def filter_adapted_algorithms(cls, problem: Any) -> list[str]:
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
