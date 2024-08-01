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
"""Base class for libraries of DOEs."""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from functools import singledispatchmethod
from multiprocessing import RLock
from multiprocessing import current_process
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import dtype
from numpy import hstack
from numpy import int32
from numpy import where

from gemseo.algos.base_driver_library import BaseDriverLibrary
from gemseo.algos.base_driver_library import DriverDescription
from gemseo.algos.base_driver_library import DriverLibraryOptionType
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.evaluation_problem import EvaluationType
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.parallel_execution.callable_parallel_execution import SUBPROCESS_NAME
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.core.serializable import Serializable
from gemseo.utils.locks import synchronized
from gemseo.utils.seeder import Seeder

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.algos.optimization_result import OptimizationResult
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


CallbackType = Callable[[int, EvaluationType], Any]
"""The type of a callback function in the context of a ."""


@dataclass
class DOEAlgorithmDescription(DriverDescription):
    """The description of a DOE algorithm."""

    handle_integer_variables: bool = True

    minimum_dimension: int = 1
    """The minimum dimension of the parameter space."""


class BaseDOELibrary(BaseDriverLibrary, Serializable):
    """Base class for libraries of DOEs."""

    samples: RealArray
    """The design vector samples in the design space.

    The design space variable types stored as dtype metadata.

    To access those in the unit hypercube,
    use :attr:`.unit_samples`.
    """

    unit_samples: RealArray
    """The design vector samples projected in the unit hypercube.

    In the case of a design space of dimension :math:`d`,
    the unit hypercube is :math:`[0,1]^d`.

    To access those in the design space,
    use :attr:`.samples`.
    """

    lock: RLock
    """The lock protecting database storage in multiprocessing."""

    _N_SAMPLES: Final[str] = "n_samples"
    _SEED: Final[str] = "seed"

    _seeder: Seeder
    """A seed generator."""

    _USE_UNIT_HYPERCUBE: ClassVar[bool] = True
    """Whether the algorithms use a unit hypercube to generate the design samples."""

    _NORMALIZE_DS = False

    __compute_jacobians: bool
    """Whether to compute the Jacobians."""

    __output_functions: list[MDOFunction]
    """The functions to compute the outputs."""

    __jacobian_functions: list[MDOFunction]
    """The functions to compute the Jacobians."""

    # TODO: use DesignSpace enum once there are hashable.
    __DESIGN_VARIABLE_TYPE_TO_PYTHON_TYPE: Final[dict[str, type]] = {
        "float": float,
        "integer": int32,
    }
    _ATTR_NOT_TO_SERIALIZE: ClassVar[set[str]] = {"lock"}

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self.samples = array([])
        self.unit_samples = array([])
        self._seeder = Seeder()
        self.__compute_jacobians = False
        self.__output_functions = []
        self.__jacobian_functions = []
        self.lock = RLock()

    def _init_shared_memory_attrs_after(self) -> None:
        self.lock = RLock()

    @property
    def seed(self) -> int:
        """The default seed used for reproducibility reasons."""
        return self._seeder.default_seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seeder.default_seed = value

    def _pre_run(
        self,
        problem: OptimizationProblem,
        **options: DriverLibraryOptionType,
    ) -> None:
        design_space = problem.design_space
        self.__check_unnormalization_capability(design_space)
        super()._pre_run(problem, **options)
        problem.stop_if_nan = False
        self.unit_samples = self._generate_unit_samples(design_space, **options)
        LOGGER.debug(
            (
                "The DOE algorithm %s of %s has generated %s samples "
                "in the input unit hypercube of dimension %s."
            ),
            self._algo_name,
            self.__class__.__name__,
            *self.unit_samples.shape,
        )
        self.samples = self.__convert_unit_samples_to_samples(problem)
        self._init_iter_observer(problem, len(self.unit_samples))

    def __convert_unit_samples_to_samples(
        self, problem: OptimizationProblem
    ) -> RealArray:
        """Convert the unit design vector samples to design vector samples.

        We also set the design variable types as dtype metadata.

        Args:
            problem: The problem to be solved.

        Returns:
            The design vector samples.
        """
        design_space = problem.design_space
        samples = design_space.untransform_vect(self.unit_samples, no_check=True)
        variable_types = design_space.variable_types
        unique_variable_types = {t[0] for t in variable_types.values()}
        if len(unique_variable_types) > 1:
            # When the design space have both float and integer variables,
            # the samples array has the float dtype.
            # We record the integer variables types to later be able to restore the
            # proper data type.
            python_var_types = {
                name: self.__DESIGN_VARIABLE_TYPE_TO_PYTHON_TYPE[type_[0]]
                for name, type_ in variable_types.items()
                if type_[0] != DesignSpace.DesignVariableType.FLOAT
            }
            samples.dtype = dtype(samples.dtype, metadata=python_var_types)

        return samples

    @abstractmethod
    def _generate_unit_samples(
        self, design_space: DesignSpace, **options: Any
    ) -> RealArray:
        """Generate the samples of the design vector in the unit hypercube.

        Args:
            design_space: The design space to be sampled.
            **options: The options of the DOE algorithm.

        Returns:
            The samples of the design vector in the unit hypercube.
        """

    def _run(
        self,
        problem: OptimizationProblem,
        eval_jac: bool = False,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
        use_database: bool = True,
        callbacks: Iterable[CallbackType] = (),
        **options: Any,
    ) -> OptimizationResult:
        """
        Args:
            eval_jac: Whether to evaluate the Jacobian function.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The time to wait between each sample
                evaluation, in seconds.
            use_database: Whether to store the evaluations in the database.
            callbacks: The functions to be evaluated
                after each call to :meth:`.OptimizationProblem.evaluate_functions`;
                to be called as ``callback(index, (output, jacobian))``.
            **options: These options are not used.

        Warnings:
            This class relies on multiprocessing features when ``n_processes > 1``,
            it is therefore necessary to protect its execution with an
            ``if __name__ == '__main__':`` statement when working on Windows.
        """  # noqa: D205, D212
        self.__compute_jacobians = eval_jac
        self.__output_functions, self.__jacobian_functions = self.problem.get_functions(
            jacobian_names=() if self.__compute_jacobians else None,
            observable_names=(),
        )
        callbacks = list(callbacks)
        if n_processes > 1:
            LOGGER.info("Running DOE in parallel on n_processes = %s", n_processes)
            # Given a ndarray input value,
            # the worker evaluates the functions attached to the problem
            # with up to n_processes simultaneous processes.
            parallel = CallableParallelExecution(
                [self._worker],
                n_processes=n_processes,
                wait_time_between_fork=wait_time_between_samples,
            )
            database = problem.database
            if use_database:
                # Add a callback to store the samples in the database on the fly.
                callbacks.append(self.__store_in_database)
                # Initialize the order of samples
                # as parallel execution does not guarantee it.
                for sample in self.samples:
                    database.store(sample, {})

            # The list of inputs of the tasks is the list of samples
            # A callback function stores the samples on the fly
            # during the parallel execution.
            parallel.execute(self.samples, exec_callback=callbacks)
            if use_database:
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
            for index, input_data in enumerate(self.samples):
                try:
                    output_data, jacobian_data = problem.evaluate_functions(
                        design_vector=input_data,
                        preprocess_design_vector=False,
                        design_vector_is_normalized=False,
                        output_functions=self.__output_functions,
                        jacobian_functions=self.__jacobian_functions,
                    )
                    for callback in callbacks:
                        callback(index, (output_data, jacobian_data))
                except ValueError:  # noqa: PERF203
                    LOGGER.exception(
                        "Problem with evaluation of sample:"
                        "%s result is not taken into account in DOE.",
                        input_data,
                    )

        return self._get_optimum_from_database(problem)

    def _worker(self, sample: RealArray) -> EvaluationType:
        """Wrap the evaluation of the functions for parallel execution.

        Args:
            sample: A point from the design space.

        Returns:
            The computed values.
        """
        if current_process().name == SUBPROCESS_NAME:
            self._deactivate_progress_bar()
            self.problem.database.clear_listeners()

        return self.problem.evaluate_functions(
            design_vector=sample,
            preprocess_design_vector=False,
            design_vector_is_normalized=False,
            output_functions=self.__output_functions,
            jacobian_functions=self.__jacobian_functions,
        )

    @synchronized
    def __store_in_database(
        self,
        index: int,
        output_and_jacobian_data: EvaluationType,
    ) -> None:
        """Store the output and Jacobian data in the database.

        Args:
            index: The sample index.
            output_and_jacobian_data: The output and Jacobian data.
        """
        data, jacobian_data = output_and_jacobian_data
        if jacobian_data:
            for output_name, jacobian in jacobian_data.items():
                data[self.problem.database.get_gradient_name(output_name)] = jacobian

        self.problem.database.store(self.samples[index], data)

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
            msg = f"The components {components} of the design space are unbounded."
            raise ValueError(msg)

    def compute_doe(
        self,
        variables_space: DesignSpace | int,
        n_samples: int | None = None,
        unit_sampling: bool = False,
        **options: DriverLibraryOptionType,
    ) -> RealArray:
        """Compute a design of experiments (DOE) in a variables space.

        Args:
            variables_space: Either the variables space to be sampled or its dimension.
            n_samples: The number of samples.
                If ``None``,
                it is deduced from the ``variables_spaces`` and the ``options``.
            unit_sampling: Whether to sample in the unit hypercube.
                If the value provided in ``variables_space`` is the dimension,
                the samples will be generated in the unit hypercube
                whatever the value of ``unit_sampling``.
            **options: The options of the DOE algorithm.

        Returns:
            The design of experiments
            whose rows are the samples and columns the variables.
        """
        design_space = self.__get_design_space(variables_space)
        if not unit_sampling:
            self.__check_unnormalization_capability(design_space)

        self._init_options_grammar()
        if self._N_SAMPLES in self._option_grammar:
            options[self._N_SAMPLES] = n_samples

        unit_samples = self._generate_unit_samples(
            design_space,
            **self._update_algorithm_options(
                initialize_options_grammar=False, **options
            ),
        )
        if unit_sampling:
            return unit_samples

        return design_space.untransform_vect(unit_samples, no_check=True)

    @singledispatchmethod
    def __get_design_space(self, design_space):
        """Return a design space.

        Args:
            design_space: Either a design space or a design space dimension.

        Returns:
            A design space.
        """
        return design_space

    @__get_design_space.register
    def _(self, design_space: DesignSpace):
        """Return a design space.

        Args:
            design_space: A design space

        Returns:
            The design space passed as argument.
        """
        return design_space

    @__get_design_space.register
    def _(self, design_space: int):
        """Return a design space from a design space dimension.

        Args:
            design_space: A design space dimension.

        Returns:
            A design space
            containing a single variable called ``"x"``
            whose size is the dimension passed as argument
            and lower and upper bounds are 0 and 1 respectively.
        """
        design_space_ = DesignSpace()
        design_space_.add_variable("x", size=design_space, l_b=0.0, u_b=1.0)
        return design_space_
