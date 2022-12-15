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
import traceback
from dataclasses import dataclass
from multiprocessing import current_process
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import Tuple
from typing import Union

from numpy import array
from numpy import ndarray
from numpy import savetxt

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_quality import compute_phip_criterion
from gemseo.algos.driver_lib import DriverDescription
from gemseo.algos.driver_lib import DriverLib
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.parallel_execution import ParallelExecution
from gemseo.core.parallel_execution import SUBPROCESS_NAME

LOGGER = logging.getLogger(__name__)

DOELibraryOptionType = Union[str, float, int, bool, List[str], ndarray]
DOELibraryOutputType = Tuple[Dict[str, Union[float, ndarray]], Dict[str, ndarray]]


@dataclass
class DOEAlgorithmDescription(DriverDescription):
    """The description of a DOE algorithm."""

    handle_integer_variables: bool = True

    minimum_dimension: int = 1
    """The minimum dimension of the parameter space."""


class DOELibrary(DriverLib):
    """Abstract class to use for DOE library link See DriverLib."""

    unit_samples: ndarray
    """The input samples transformed in :math:`[0,1]`."""

    samples: ndarray
    """The input samples."""

    seed: int
    """The seed to be used for replicability reasons.

    It increments with each generation of samples
    so that repeating the generation of sets of :math:`N` leads to different sets.
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
    _VARIABLES_NAMES = "variables_names"
    _VARIABLES_SIZES = "variables_sizes"
    SEED = "seed"
    _NORMALIZE_DS = False

    def __init__(self):
        """Constructor Abstract class."""
        super().__init__()
        self.unit_samples = array([])
        self.samples = array([])
        self.seed = 0
        self.eval_jac = False

    # TODO: API: remove this method
    @staticmethod
    def compute_phip_criteria(samples: ndarray, power: float = 10.0) -> float:
        r"""Compute the :math:`\phi^p` space-filling criterion (the smaller the better).

        See :cite:`morris1995`.

        Args:
            samples: The samples of the input variables.
            power: The power :math:`p` of the :math:`\phi^p` criterion.

        Returns:
            The :math:`\phi^p` space-filling criterion.
        """
        return compute_phip_criterion(samples, power)

    def _pre_run(
        self,
        problem: OptimizationProblem,
        algo_name: str,
        **options: DOELibraryOptionType,
    ) -> None:
        super()._pre_run(problem, algo_name, **options)
        problem.stop_if_nan = False
        options[self.DIMENSION] = self.problem.dimension
        options[self._VARIABLES_NAMES] = self.problem.design_space.variables_names
        options[self._VARIABLES_SIZES] = self.problem.design_space.variables_sizes
        self.unit_samples = self._generate_samples(**options)
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
        self.samples = self.problem.design_space.untransform_vect(
            self.unit_samples, no_check=True
        )

        if options.get(self.N_PROCESSES, 1) > 1:
            # Initialize the order as it is not necessarily guaranteed
            # when using parallel execution.
            for sample in self.samples:
                self.problem.database.store(sample, {}, add_iter=True)

        self.init_iter_observer(len(self.unit_samples))

    def _generate_samples(self, **options: Any) -> ndarray:
        """Generate the samples of the input variables.

        Args:
            **options: The options of the DOE algorithm.
        """
        raise NotImplementedError()

    def __call__(self, n_samples: int, dimension: int, **options: Any) -> ndarray:
        """Generate a design of experiments in the unit hypercube.

        Args:
            n_samples: The number of samples.
            dimension: The dimension of the input space.
            **options: The options of the DOE algorithm.

        Returns:
            A design of experiments in the unit hypercube.
        """
        return self._generate_samples(
            **self.__get_algorithm_options(options, n_samples, dimension)
        )

    def _run(self, **options: Any) -> OptimizationResult:
        eval_jac = options.get(self.EVAL_JAC, False)
        n_processes = options.get(self.N_PROCESSES, 1)
        wait_time_between_samples = options.get(self.WAIT_TIME_BETWEEN_SAMPLES, 0)
        self.evaluate_samples(eval_jac, n_processes, wait_time_between_samples)
        return self.get_optimum_from_database()

    def _generate_fullfact(
        self,
        dimension: int,
        n_samples: int | None = None,
        levels: int | Iterable[int] | None = None,
    ) -> ndarray:
        """Generate a full-factorial DOE.

        Generate a full-factorial DOE based on either the number of samples,
        or the number of levels per input direction.
        When the number of samples is prescribed,
        the levels are deduced and are uniformly distributed among all the inputs.

        Args:
            dimension: The dimension of the parameter space.
            n_samples: The maximum number of samples from which the number of levels
                per input is deduced.
                The number of samples which is finally applied
                is the product of the numbers of levels.
                If ``None``, the algorithm uses the number of levels per input dimension
                provided by the argument ``levels``.
            levels: The number of levels per input direction.
                If ``levels`` is given as a scalar value, the same number of
                levels is used for all the inputs.
                If ``None``, the number of samples provided in argument ``n_samples``
                is used in order to deduce the levels.

        Returns:
            The values of the DOE.

        Raises:
            ValueError:
                * If neither ``n_samples`` nor ``levels`` is provided.
                * If both ``n_samples`` and ``levels`` are provided.
        """
        if not levels and not n_samples:
            raise ValueError(
                "Either 'n_samples' or 'levels' is required as an input "
                "parameter for the full-factorial DOE."
            )
        if levels and n_samples:
            raise ValueError(
                "Only one input parameter among 'n_samples' and 'levels' "
                "must be given for the full-factorial DOE."
            )

        if n_samples is not None:
            levels = self._compute_fullfact_levels(n_samples, dimension)

        if isinstance(levels, int):
            levels = [levels] * dimension

        return self._generate_fullfact_from_levels(levels)

    def _generate_fullfact_from_levels(self, levels) -> ndarray:  # Iterable[int]
        """Generate the full-factorial DOE from levels per input direction.

        Args:
            levels: The number of levels per input direction.

        Returns:
            The values of the DOE.
        """
        raise NotImplementedError()

    @staticmethod
    def _compute_fullfact_levels(n_samples: int, dimension: int) -> list[int]:
        """Compute the number of levels per input dimension for a full factorial design.

        Args:
            n_samples: The number of samples.
            dimension: The dimension of the input space.

        Returns:
            The number of levels per input dimension.
        """
        n_samples_dir = int(n_samples ** (1.0 / dimension))
        final_n_samples = n_samples_dir**dimension
        if final_n_samples != n_samples:
            LOGGER.warning(
                (
                    "A full-factorial DOE of %s samples in dimension %s does not exist;"
                    " use %s samples instead, "
                    "i.e. the largest %s-th integer power less than %s."
                ),
                n_samples,
                dimension,
                final_n_samples,
                dimension,
                n_samples,
            )
        return [n_samples_dir] * dimension

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
    ):
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
            parallel = ParallelExecution(self._worker, n_processes=n_processes)
            parallel.wait_time_between_fork = wait_time_between_samples
            # Define a callback function to store the samples on the fly
            # during the parallel execution
            database = self.problem.database

            # Initialize the order as it is not necessarily guaranteed
            # when using parallel execution
            for sample in self.samples:
                database.store(sample, {}, add_iter=True)

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
                        out["@" + key] = val

                database.store(self.samples[index], out)

            # The list of inputs of the tasks is the list of samples
            parallel.execute(self.unit_samples, exec_callback=store_callback)
            # We added empty entries by default to keep order in the database
            # but when the DOE point is failed, this is not consistent
            # with the serial exec, so we clean the DB
            database.remove_empty_entries()

        else:  # Sequential execution
            if wait_time_between_samples != 0:
                LOGGER.warning(
                    "Wait time between samples option is ignored" " in sequential run."
                )
            for sample in self.samples:
                try:
                    self.problem.evaluate_functions(
                        x_vect=sample,
                        eval_jac=self.eval_jac,
                        normalize=False,
                        eval_observables=True,
                    )
                except ValueError:
                    LOGGER.error(
                        "Problem with evaluation of sample :"
                        "%s result is not taken into account "
                        "in DOE.",
                        str(sample),
                    )
                    LOGGER.error(traceback.format_exc())

    @staticmethod
    def _rescale_samples(samples):
        """When the samples are out of the [0, 1] bounds, rescales them.

        Args:
            samples: The samples to rescale.

        Returns:
            The samples scaled in the interval [0, 1].
        """
        if (not (samples >= 0.0).all()) or (not (samples <= 1.0).all()):
            max_s = samples.max()
            min_s = samples.min()
            if abs(max_s - min_s) > 1e-14:
                samples_n = (samples - min_s) / (max_s - min_s)
                assert samples_n.shape == samples.shape
                return samples_n
            return samples
        return samples

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
        options = self.__get_algorithm_options(options, size, variables_space.dimension)
        options[self._VARIABLES_NAMES] = variables_space.variables_names
        options[self._VARIABLES_SIZES] = variables_space.variables_sizes
        doe = self._generate_samples(**options)
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
