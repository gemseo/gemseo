# -*- coding: utf-8 -*-
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
"""
DOE library base class wrapper
******************************
"""

from __future__ import division, unicode_literals

import logging
import traceback
from multiprocessing import current_process
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import six
from custom_inherit import DocInheritMeta
from numpy import ndarray, savetxt
from scipy.spatial import distance

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.driver_lib import DriverLib
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.parallel_execution import SUBPROCESS_NAME, ParallelExecution

LOGGER = logging.getLogger(__name__)

DOELibraryOptionType = Union[str, float, int, bool, List[str], ndarray]
DOELibraryOutputType = Tuple[Dict[str, Union[float, ndarray]], Dict[str, ndarray]]


@six.add_metaclass(
    DocInheritMeta(
        abstract_base_class=True,
        style="google_with_merge",
        include_special_methods=True,
    )
)
class DOELibrary(DriverLib):
    """Abstract class to use for DOE library link See DriverLib."""

    MIN_DIMS = "min_dims"
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

    def __init__(self):
        """Constructor Abstract class."""
        super(DOELibrary, self).__init__()
        self.samples = None
        self.seed = 0

    @staticmethod
    def compute_phip_criteria(samples, power=10.0):
        r"""Compute the :math:`\phi^p` space-filling criterion.

        See Morris & Mitchell, Exploratory designs for computational experiments, 1995.

        :param samples: design variables list
        :param power: The power p of the :math:`\phi^p` criteria.
        """

        def compute_distance(sample, other_sample):
            r"""Compute the distance used by the :math:`\phi^p` criterion.

            :param sample: A sample.
            :param other_sample: Another sample.
            """
            return sum(abs(sample - other_sample)) ** (-power)

        criterion = sum(distance.pdist(samples, compute_distance)) ** (1.0 / power)
        LOGGER.info(
            "Value of Phi^p criterion with p=%s (Morris & Mitchell, 1995): %s",
            power,
            criterion,
        )
        return criterion

    def _pre_run(
        self,
        problem,  # type: OptimizationProblem
        algo_name,  # type: str
        **options  # type: DOELibraryOptionType
    ):  # type: (...) -> None
        super(DOELibrary, self)._pre_run(problem, algo_name, **options)

        problem.stop_if_nan = False
        LOGGER.info("%s", problem)
        options[self.DIMENSION] = self.problem.dimension
        options[self._VARIABLES_NAMES] = self.problem.design_space.variables_names
        options[self._VARIABLES_SIZES] = self.problem.design_space.variables_sizes
        self.samples = self._generate_samples(**options)

        # Initialize the order as it is not necessarily guaranteed
        # when using parallel execution.
        unnormalize_vect = self.problem.design_space.unnormalize_vect
        round_vect = self.problem.design_space.round_vect
        for sample in self.samples:
            self.problem.database.store(
                round_vect(unnormalize_vect(sample)), {}, add_iter=True
            )

        self.init_iter_observer(len(self.samples), "DOE sampling")
        self.problem.add_callback(self.new_iteration_callback)

    def _generate_samples(self, **options):
        """Generates the list of x samples.

        :param options: the options dict for the algorithm,
               see associated JSON file
        """
        raise NotImplementedError()

    def __call__(self, n_samples, dimension, **options):
        """Generate samples in the unit hypercube.

        :param int n_samples: number of samples.
        :param int dimension: parameter space dimension.
        :param options: options passed to the DOE algorithm.
        :returns: samples.
        :rtype: ndarray
        """
        options = self.__get_algorithm_options(options, n_samples, dimension)
        return self._generate_samples(**options)

    def _run(self, **options):
        """Runs the algorithm, to be overloaded by subclasses.

        :param options: the options dict for the algorithm,
            see associated JSON file
        """
        eval_jac = options.get(self.EVAL_JAC, False)
        n_processes = options.get(self.N_PROCESSES, 1)
        wait_time_between_samples = options.get(self.WAIT_TIME_BETWEEN_SAMPLES, 0)
        self.evaluate_samples(eval_jac, n_processes, wait_time_between_samples)
        return self.get_optimum_from_database()

    def _generate_fullfact(
        self,
        dimension,
        n_samples=None,  # type: Optional[int]
        levels=None,  # type: Optional[Union[int, Iterable[int]]]
    ):  # type: (...) -> ndarray
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

    def _generate_fullfact_from_levels(
        self, levels  # Iterable[int]
    ):  # type: (...) -> ndarray
        """Generate the full-factorial DOE from levels per input direction.

        Args:
            levels: The number of levels per input direction.

        Returns:
            The values of the DOE.
        """
        raise NotImplementedError()

    def _compute_fullfact_levels(self, n_samples, dimension):
        """Compute the number of levels per input dimension for a full factorial design.

        :param n_samples: number of samples
        :param int dimension: parameter space dimension.
        :returns: The number of levels per input dimension.
        :rtype: List[int]
        """
        n_samples_dir = int(n_samples ** (1.0 / dimension))
        LOGGER.info(
            "Full factorial design required. Number of samples along each"
            " direction for a design vector of size %s with %s samples: %s",
            str(dimension),
            str(n_samples),
            str(n_samples_dir),
        )
        LOGGER.info(
            "Final number of samples for DOE = %s vs %s requested",
            str(n_samples_dir ** dimension),
            str(n_samples),
        )
        return [n_samples_dir] * dimension

    def export_samples(self, doe_output_file):
        """Export samples generated by DOE library to a csv file.

        :param doe_output_file: export file name
        :type doe_output_file: string
        """
        if self.samples is None:
            raise RuntimeError("Samples are None, execute method before export.")
        savetxt(doe_output_file, self.samples, delimiter=",")

    def _worker(
        self, sample  # type: ndarray
    ):  # type: (...) -> DOELibraryOutputType
        """Wrap the evaluation of the functions for parallel execution.

        Args:
            sample: The values for the evaluation of the functions.

        Returns:
            The computed values.
        """
        if current_process().name == SUBPROCESS_NAME:
            self.deactivate_progress_bar()
            self.problem.database.clear_listeners()
        return self.problem.evaluate_functions(sample, self.eval_jac)

    def evaluate_samples(
        self,
        eval_jac=False,  # type: bool
        n_processes=1,  # type: int
        wait_time_between_samples=0.0,  # type: float
    ):
        """Evaluate all the functions of the optimization problem at the samples.

        Args:
            eval_jac: Whether to evaluate the jacobian.
            n_processes: The number of processes used to evaluate the samples.
            wait_time_between_samples: The time to wait between each sample
                evaluation, in seconds.
        """
        self.eval_jac = eval_jac
        sample_to_design = self.problem.design_space.untransform_vect
        unnormalize_grad = self.problem.design_space.unnormalize_grad
        round_vect = self.problem.design_space.round_vect
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
                x_u = sample_to_design(sample)
                x_r = round_vect(x_u)
                database.store(x_r, {}, add_iter=True)

            def store_callback(
                index,  # type: int
                outputs,  # type: DOELibraryOutputType
            ):  # type: (...) -> None
                """Store the outputs in the database.

                Args:
                    index: The sample index.
                    outputs: The outputs of the parallel execution.
                """
                out, jac = outputs
                if jac:
                    for key, val in jac.items():
                        val = unnormalize_grad(val)
                        out["@" + key] = val
                x_u = sample_to_design(self.samples[index])
                x_r = round_vect(x_u)
                database.store(x_r, out)

            # The list of inputs of the tasks is the list of samples
            parallel.execute(self.samples, exec_callback=store_callback)
            # We added empty entries by default to keep order in the database
            # but when the DOE point is failed, this is not consistent
            # with the serial exec, so we clean the DB
            database.remove_empty_entries()

        else:  # Sequential execution
            if wait_time_between_samples != 0:
                LOGGER.warning(
                    "Wait time between samples option is ignored" " in sequential run."
                )
            for x_norm in self.samples:
                try:
                    self.problem.evaluate_functions(x_norm, eval_jac)
                except ValueError:
                    LOGGER.error(
                        "Problem with evaluation of sample :"
                        "%s result is not taken into account "
                        "in DOE.",
                        str(x_norm),
                    )
                    LOGGER.error(traceback.format_exc())

    @staticmethod
    def is_algorithm_suited(algo_dict, problem):
        """Checks if the algorithm is suited to the problem according to its algo dict.

        :param algo_dict: the algorithm characteristics
        :param problem: the opt_problem to be solved
        """
        return True

    @staticmethod
    def _rescale_samples(samples):
        """When the samples are out of the [0,1] bounds, rescales them.

        :param samples: the samples to rescale
        :returns: samples normed ndarray
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
        variables_space,  # type: DesignSpace
        size=None,  # type: Optional[int]
        unit_sampling=False,  # type: bool
        **options  # type: DOELibraryOptionType
    ):  # type: (...) -> ndarray
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

        return variables_space.untransform_vect(doe)

    def __get_algorithm_options(
        self,
        options,  # type: Mapping[str, DOELibraryOptionType]
        size,  # type: Optional[int]
        dimension,  # type: int
    ):  # type: (...) -> Dict[str,DOELibraryOptionType]
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
