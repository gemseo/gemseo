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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base DOE library."""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from multiprocessing import current_process
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Union

from numpy import array
from numpy import dtype
from numpy import hstack
from numpy import int32
from numpy import ndarray
from numpy import savetxt
from numpy import where

from gemseo import SEED
from gemseo.algos.driver_library import DriverDescription
from gemseo.algos.driver_library import DriverLibrary
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.parallel_execution.callable_parallel_execution import SUBPROCESS_NAME
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path

    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.opt_problem import OptimizationProblem
    from gemseo.algos.opt_result import OptimizationResult

LOGGER = logging.getLogger(__name__)

DOELibraryOptionType = Union[str, float, int, bool, list[str], ndarray]
DOELibraryOutputType = tuple[dict[str, Union[float, ndarray]], dict[str, ndarray]]


@dataclass
class DOEAlgorithmDescription(DriverDescription):
    """The description of a DOE algorithm."""

    handle_integer_variables: bool = True

    minimum_dimension: int = 1
    """The minimum dimension of the parameter space."""


class DOELibrary(DriverLibrary):
    """Abstract class to use for DOE library link See DriverLibrary."""

    unit_samples: ndarray
    """The input samples transformed in :math:`[0,1]`."""

    samples: ndarray
    """The input samples with the design space variable types stored as dtype
    metadata."""

    seed: int
    """The seed to be used for reproducibility reasons.

    This seed is initialized at 0 and each call to :meth:`.execute` increments it before
    using it.
    """

    eval_jac: bool
    """Whether to evaluate the Jacobian."""

    DESIGN_ALGO_NAME = "Design algorithm"
    SAMPLES_TAG = "samples"
    PHIP_CRITERIA = "phi^p"
    N_SAMPLES = "n_samples"
    LEVEL_KEYWORD = "levels"
    EVAL_JAC = "eval_jac"
    N_PROCESSES = "n_processes"
    WAIT_TIME_BETWEEN_SAMPLES = "wait_time_between_samples"
    DIMENSION = "dimension"
    _VARIABLE_NAMES = "variable_names"
    _VARIABLE_SIZES = "variable_sizes"
    SEED = "seed"
    _NORMALIZE_DS = False

    # TODO: use DesignSpace enum once there are hashable.
    __DESIGN_VARIABLE_TYPE_TO_PYTHON_TYPE: Final[dict[str, type]] = {
        "float": float,
        "integer": int32,
    }

    _USE_UNIT_HYPERCUBE: ClassVar[bool] = True
    """Whether the algorithms use a unit hypercube to generate the input samples."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.unit_samples = array([])
        self.samples = array([])
        self.seed = SEED
        self.eval_jac = False

    def _pre_run(
        self,
        problem: OptimizationProblem,
        algo_name: str,
        **options: DOELibraryOptionType,
    ) -> None:
        self.__check_unnormalization_capability(self.problem.design_space)
        super()._pre_run(problem, algo_name, **options)
        problem.stop_if_nan = False
        options[self.DIMENSION] = self.problem.dimension
        options[self._VARIABLE_NAMES] = self.problem.design_space.variable_names
        options[self._VARIABLE_SIZES] = self.problem.design_space.variable_sizes

        self.unit_samples = self.__generate_samples(**options)

        LOGGER.debug(
            (
                "The DOE algorithm %s of %s has generated %s samples "
                "in the input unit hypercube of dimension %s."
            ),
            self.algo_name,
            self.__class__.__name__,
            len(self.unit_samples),
            self.unit_samples.shape[1],
        )

        self.samples = self.__create_samples()

        if options.get(self.N_PROCESSES, 1) > 1:
            # Initialize the order as it is not necessarily guaranteed
            # when using parallel execution.
            for sample in self.samples:
                self.problem.database.store(sample, {})

        self.init_iter_observer(len(self.unit_samples))

    def __create_samples(self) -> ndarray:
        """Create the samples with the design variable types as dtype metadata.

        Returns:
            The samples.
        """
        samples = self.problem.design_space.untransform_vect(
            self.unit_samples, no_check=True
        )

        variable_types = self.problem.design_space.variable_types
        unique_variable_types = {t[0] for t in variable_types.values()}

        if len(unique_variable_types) > 1:
            # When the design space have both float and integer variables,
            # the samples array has the float dtype.
            # We record the integer variables types to later be able to restore the
            # proper data type.
            python_var_types = {
                name: self.__DESIGN_VARIABLE_TYPE_TO_PYTHON_TYPE[type_[0]]
                for name, type_ in variable_types.items()
                if type_[0] != "float"
            }
            samples.dtype = dtype(samples.dtype, metadata=python_var_types)

        return samples

    def __generate_samples(self, **options: Any) -> ndarray:
        """Generate the samples of the input variables.

        Args:
            **options: The options of the DOE algorithm.
        """
        self.seed += 1
        return self._generate_samples(**options)

    def _get_seed(self, seed: int | None) -> int:
        """Return a seed for the random number generator.

        Args:
            seed: A seed if any.

        Returns:
            The seed for the random number generator.
        """
        return self.seed if seed is None else seed

    @abstractmethod
    def _generate_samples(self, **options: Any) -> ndarray:
        """Generate the samples of the input variables.

        Args:
            **options: The options of the DOE algorithm.
        """

    def __call__(
        self, n_samples: int | None, dimension: int, **options: Any
    ) -> ndarray:
        """Generate a design of experiments in the unit hypercube.

        Args:
            n_samples: The number of samples.
                If ``None``, the number of samples is deduced from the ``options``.
            dimension: The dimension of the input space.
            **options: The options of the DOE algorithm.

        Returns:
            A design of experiments in the unit hypercube.
        """
        return self.__generate_samples(
            **self.__get_algorithm_options(options, n_samples, dimension)
        )

    def _run(self, **options: Any) -> OptimizationResult:
        eval_jac = options.get(self.EVAL_JAC, False)
        n_processes = options.get(self.N_PROCESSES, 1)
        wait_time_between_samples = options.get(self.WAIT_TIME_BETWEEN_SAMPLES, 0)
        self.evaluate_samples(eval_jac, n_processes, wait_time_between_samples)
        return self.get_optimum_from_database()

    def export_samples(self, doe_output_file: Path | str) -> None:
        """Export the samples generated by DOE library to a CSV file.

        Args:
            doe_output_file: The path to the output file.
        """
        if not self.unit_samples.size:
            raise RuntimeError("Samples are missing, execute method before export.")

        savetxt(doe_output_file, self.unit_samples, delimiter=",")

    def _worker(self, sample: ndarray) -> DOELibraryOutputType:
        """Wrap the evaluation of the functions for parallel execution.

        Args:
            sample: A point from the unit hypercube.

        Returns:
            The computed values.
        """
        if current_process().name == SUBPROCESS_NAME:
            self.deactivate_progress_bar()
            self.problem.database.clear_listeners()

        return self.problem.evaluate_functions(
            x_vect=self.problem.design_space.untransform_vect(sample, no_check=True),
            eval_jac=self.eval_jac,
            eval_observables=True,
            normalize=False,
        )

    def evaluate_samples(
        self,
        eval_jac: bool = False,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
    ) -> None:
        """Evaluate all the functions of the optimization problem at the samples.

        Args:
            eval_jac: Whether to evaluate the Jacobian.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The time to wait between each sample
                evaluation, in seconds.

        Warnings:
            This class relies on multiprocessing features when ``n_processes > 1``,
            it is therefore necessary to protect its execution with an
            ``if __name__ == '__main__':`` statement when working on Windows.
        """
        self.eval_jac = eval_jac
        if n_processes > 1:
            LOGGER.info("Running DOE in parallel on n_processes = %s", str(n_processes))
            # Create a list of tasks: execute functions
            parallel = CallableParallelExecution(
                [self._worker], n_processes=n_processes
            )
            parallel.wait_time_between_fork = wait_time_between_samples
            # Define a callback function to store the samples on the fly
            # during the parallel execution
            database = self.problem.database

            # Initialize the order as it is not necessarily guaranteed
            # when using parallel execution
            for sample in self.samples:
                database.store(sample, {})

            def store_callback(
                index: int,
                outputs: DOELibraryOutputType,
            ) -> None:
                """Store the outputs in the database.

                Args:
                    index: The sample index.
                    outputs: The outputs of the parallel execution.
                """
                out, jac = outputs
                if jac:
                    for key, val in jac.items():
                        out[database.get_gradient_name(key)] = val

                database.store(self.samples[index], out)

            # The list of inputs of the tasks is the list of samples
            parallel.execute(self.unit_samples, exec_callback=store_callback)
            # We added empty entries by default to keep order in the database
            # but when the DOE point is failed, this is not consistent
            # with the serial exec, so we clean the DB
            database.remove_empty_entries()

        else:
            # Sequential execution
            if wait_time_between_samples != 0:
                LOGGER.warning(
                    "Wait time between samples option is ignored in sequential run."
                )
            for sample in self.samples:
                try:
                    self.problem.evaluate_functions(
                        x_vect=sample,
                        eval_jac=self.eval_jac,
                        normalize=False,
                    )
                except ValueError:  # noqa: PERF203
                    LOGGER.exception(
                        "Problem with evaluation of sample :"
                        "%s result is not taken into account "
                        "in DOE.",
                        sample,
                    )

    @classmethod
    def __check_unnormalization_capability(cls, design_space) -> None:
        """Check if a point of the unit hypercube can be unnormalized.

        Args:
            design_space: The design space to unnormalize the point.

        Raises:
            ValueError: When some components of the design space are unbounded.
        """
        if not cls._USE_UNIT_HYPERCUBE or isinstance(design_space, ParameterSpace):
            return

        components = set(where(hstack(list(design_space.normalize.values())) == 0)[0])
        if components:
            raise ValueError(
                f"The components {components} of the design space are unbounded."
            )

    def compute_doe(
        self,
        variables_space: DesignSpace,
        size: int | None = None,
        unit_sampling: bool = False,
        **options: DOELibraryOptionType,
    ) -> ndarray:
        """Compute a design of experiments (DOE) in a variables space.

        Args:
            variables_space: The variables space to be sampled.
            size: The size of the DOE.
                If ``None``, the size is deduced from the ``options``.
            unit_sampling: Whether to sample in the unit hypercube.
            **options: The options of the DOE algorithm.

        Returns:
            The design of experiments
            whose rows are the samples and columns the variables.
        """
        if not unit_sampling:
            self.__check_unnormalization_capability(variables_space)

        options = self.__get_algorithm_options(options, size, variables_space.dimension)
        options[self._VARIABLE_NAMES] = variables_space.variable_names
        options[self._VARIABLE_SIZES] = variables_space.variable_sizes
        doe = self.__generate_samples(**options)
        if unit_sampling:
            return doe

        return variables_space.untransform_vect(doe, no_check=True)

    def __get_algorithm_options(
        self,
        options: MutableMapping[str, DOELibraryOptionType],
        size: int | None,
        dimension: int,
    ) -> dict[str, DOELibraryOptionType]:
        """Return the algorithm options from initial ones.

        Args:
            options: The user algorithm options.
            size: The number of samples.
                If ``None``, the number is deduced from the options.
            dimension: The dimension of the variables space.

        Returns:
            The algorithm options.
        """
        options[self.N_SAMPLES] = size
        options = self._update_algorithm_options(**options)
        options[self.DIMENSION] = dimension
        return options
