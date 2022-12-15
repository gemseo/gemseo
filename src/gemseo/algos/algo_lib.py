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
"""Base class for algorithm libraries."""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Mapping
from typing import MutableMapping

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import ndarray

from gemseo.algos._unsuitability_reason import _UnsuitabilityReason
from gemseo.algos.linear_solvers.linear_problem import LinearProblem
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.utils.python_compatibility import Final
from gemseo.utils.source_parsing import get_options_doc
from gemseo.utils.string_tools import pretty_str

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


class AlgoLib(metaclass=GoogleDocstringInheritanceMeta):
    """Abstract class for algorithms libraries interfaces.

    An algorithm library solves a numerical problem
    (optim, doe, linear problem) using a particular algorithm
    from a particular family of numerical methods.

    Provide the available methods in the library for the proposed
    problem to be solved.

    To integrate an optimization package, inherit from this class
    and put your module in gemseo.algos.doe or gemseo.algo.opt,
    or gemseo.algos.linear_solver packages.
    """

    descriptions: dict[str, AlgorithmDescription]
    """The description of the algorithms contained in the library."""

    algo_name: str | None
    """The name of the algorithm used currently."""

    internal_algo_name: str | None
    """The internal name of the algorithm used currently.

    It typically corresponds to the name of the algorithm in the wrapped library if any.
    """

    problem: Any | None
    """The problem to be solved."""

    opt_grammar: JSONGrammar | None
    """The grammar defining the options of the current algorithm."""

    OPTIONS_DIR: Final[str] = "options"
    """The name of the directory containing the files of the grammars of the options."""

    OPTIONS_MAP: dict[str, str] = {}
    """The names of the options in |g| mapping to those in the wrapped library."""

    LIBRARY_NAME: ClassVar[str | None] = None
    """The name of the interfaced library."""

    _COMMON_OPTIONS_GRAMMAR: ClassVar[JSONGrammar] = JSONGrammar("AlgoLibOptions")
    """The grammar defining the options common to all the algorithms of the library."""

    def __init__(self) -> None:  # noqa:D107
        # Library settings and check
        self.descriptions = {}
        self.algo_name = None
        self.internal_algo_name = None
        self.problem = None
        self.opt_grammar = None

    def init_options_grammar(
        self,
        algo_name: str,
    ) -> JSONGrammar:
        """Initialize the options' grammar.

        Args:
            algo_name: The name of the algorithm.
        """
        # Store the lib in case we rerun the same algorithm,
        # for multilevel scenarios for instance
        # This significantly speedups the process
        # because of the option grammar that is long to create
        if self.opt_grammar is not None and self.opt_grammar.name == algo_name:
            return self.opt_grammar

        library_directory = Path(inspect.getfile(self.__class__)).parent
        options_directory = library_directory / self.OPTIONS_DIR
        algo_schema_file = options_directory / "{}_options.json".format(
            algo_name.upper()
        )
        lib_schema_file = options_directory / "{}_options.json".format(
            self.__class__.__name__.upper()
        )

        if algo_schema_file.exists():
            schema_file = algo_schema_file
        elif lib_schema_file.exists():
            schema_file = lib_schema_file
        else:
            msg = (
                "Neither the options grammar file {} for the algorithm '{}' "
                "nor the options grammar file {} for the library '{}' has been found."
            ).format(
                algo_schema_file, algo_name, lib_schema_file, self.__class__.__name__
            )
            raise ValueError(msg)

        self.opt_grammar = JSONGrammar(algo_name)
        self.opt_grammar.update(self._COMMON_OPTIONS_GRAMMAR)
        self.opt_grammar.update_from_file(schema_file)
        self.opt_grammar.set_descriptions(get_options_doc(self.__class__._get_options))

        return self.opt_grammar

    @property
    def algorithms(self) -> list[str]:
        """The available algorithms."""
        return list(self.descriptions.keys())

    def _pre_run(
        self,
        problem: LinearProblem,
        algo_name: str,
        **options: Any,
    ) -> None:  # pragma: no cover
        """Save the solver options and name in the problem attributes.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            **options: The options for the algorithm, see associated JSON file.
        """

    def _post_run(
        self,
        problem: LinearProblem,
        algo_name: str,
        result: ndarray,
        **options: Any,
    ) -> None:  # pragma: no cover
        """Save the LinearProblem to the disk when required.

        If the save_when_fail option is True, save the LinearProblem to the disk when
        the system failed and print the file name in the warnings.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
            result: The result of the run, i.e. the solution.
            **options: The options for the algorithm, see associated JSON file.
        """

    def driver_has_option(self, option_name: str) -> bool:
        """Check the existence of an option.

        Args:
            option_name: The name of the option.

        Returns:
            Whether the option exists.
        """
        return option_name in self.opt_grammar

    def _process_specific_option(
        self,
        options: MutableMapping[str, Any],
        option_key: str,
    ) -> None:  # pragma: no cover
        """Preprocess the option specifically, to be overriden by subclasses.

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
            if not self.driver_has_option(option_name):
                del options[option_name]
            else:
                self._process_specific_option(options, option_name)

        try:
            self.opt_grammar.validate(options)
        except InvalidDataException:
            raise ValueError(f"Invalid options for algorithm {self.opt_grammar.name}.")

        for option_name in list(options.keys()):  # Copy keys on purpose
            lib_option_name = self.OPTIONS_MAP.get(option_name)
            # Overload with specific keys
            if lib_option_name is not None:
                options[lib_option_name] = options[option_name]
                if lib_option_name != option_name:
                    del options[option_name]

        return options

    def _check_ignored_options(self, options: Mapping[str, Any]) -> None:
        """Check that the user did not pass options that do not exist for this driver.

        Log a warning if it is the case.

        Args:
            options: The options.
        """
        for option_name in options:
            if not self.driver_has_option(option_name):
                msg = "Driver %s has no option %s, option is ignored."
                LOGGER.warning(msg, self.algo_name, option_name)

    def execute(
        self,
        problem: Any,
        algo_name: str = None,
        **options: Any,
    ) -> None:
        """Execute the driver.

        Args:
            problem: The problem to be solved.
            algo_name: The name of the algorithm.
                If ``None``, use :attr:`algo_name` attribute
                which may have been set by the factory.
            **options: The algorithm options.
        """
        self.problem = problem

        if algo_name is not None:
            self.algo_name = algo_name

        if self.algo_name is None:
            raise ValueError(
                "Algorithm name must be either passed as "
                + "argument or set by the attribute self.algo_name"
            )

        self._check_algorithm(self.algo_name, problem)
        options = self._update_algorithm_options(**options)
        self.internal_algo_name = self.descriptions[
            self.algo_name
        ].internal_algorithm_name
        problem.check()

        self._pre_run(problem, self.algo_name, **options)
        result = self._run(**options)
        self._post_run(problem, algo_name, result, **options)

        return result

    def _update_algorithm_options(self, **options: Any) -> dict[str, Any]:
        """Update the algorithm options.

        1. Load the grammar of algorithm options.
        2. Warn about the ignored initial algorithm options.
        3. Complete the initial algorithm options with the default algorithm options.

        Args:
            **options: The initial algorithm options.

        Returns:
            The updated algorithm options.
        """
        self.init_options_grammar(self.algo_name)
        self._check_ignored_options(options)
        return self._get_options(**options)

    def _get_options(self, **options: Any) -> dict[str, Any]:
        """Retrieve the options of the library.

        To be overloaded by subclasses.
        Used to define default values for options using keyword arguments.

        Args:
            **options: The options of the algorithm.

        Returns:
            The options of the algorithm.
        """
        raise NotImplementedError()

    def _run(self, **options) -> Any:
        """Run the algorithm.

        To be overloaded by subclasses.

        Args:
            **options: The options of the algorithm.

        Returns:
            The solution of the problem.
        """
        raise NotImplementedError()

    def _check_algorithm(
        self,
        algo_name: str,
        problem: Any,
    ) -> None:
        """Check that algorithm is available and adapted to the problem.

        Set the optimization library and the algorithm name according
        to the requirements of the optimization library.

        Args:
            algo_name: The name of the algorithm.
            problem: The problem to be solved.
        """
        if algo_name not in self.descriptions:
            raise KeyError(
                f"The algorithm {algo_name} is unknown; "
                f"available ones are: {pretty_str(self.descriptions.keys())}."
            )

        unsuitability_reason = self._get_unsuitability_reason(
            self.descriptions[self.algo_name], problem
        )
        if unsuitability_reason:
            raise ValueError(
                f"The algorithm {algo_name} is not adapted to the problem "
                f"because {unsuitability_reason}."
            )

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

    def filter_adapted_algorithms(self, problem: Any) -> list[str]:
        """Filter the algorithms capable of solving the problem.

        Args:
            problem: The problem to be solved.

        Returns:
            The names of the algorithms adapted to this problem.
        """
        adapted_algorithms = []
        for algo_name, algo_description in self.descriptions.items():
            if self.is_algorithm_suited(algo_description, problem):
                adapted_algorithms.append(algo_name)

        return adapted_algorithms
