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
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import RLock
from multiprocessing import parent_process
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import TypeVar

from numpy import array
from numpy import dtype

from gemseo.core.algorithm.base_driver_library import BaseDriverLibrary
from gemseo.core.algorithm.base_driver_library import DriverDescription
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.core.problem.evaluation import EvaluationType
from gemseo.core.serializable import Serializable
from gemseo.doe.core.base_doe_settings import BaseDOESettings
from gemseo.optimization.result import OptimizationResult
from gemseo.space.base import BaseVariableSpace
from gemseo.space.design import DesignSpace
from gemseo.space.variable import data_type_to_numpy_type
from gemseo.util.hashable_ndarray import HashableNdarray
from gemseo.util.lock import synchronized
from gemseo.util.pydantic import create_model
from gemseo.util.seeder import Seeder

if TYPE_CHECKING:
    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.core.problem.evaluation import EvaluationProblem
    from gemseo.util.typing import RealArray

logger = logging.getLogger(__name__)


CallbackType = Callable[[int, EvaluationType], Any]
"""The type of a callback function."""

T = TypeVar("T", bound=BaseDOESettings)


@dataclass
class DOEAlgorithmDescription(DriverDescription):
    """The description of a DOE algorithm."""

    handle_integer_variables: bool = True
    """Whether the optimization algorithm handles integer variables."""

    minimum_dimension: int = 1
    """The minimum dimension of the parameter space."""

    settings_class: type[BaseDOESettings] = BaseDOESettings
    """The Pydantic model for the DOE library settings."""


class BaseDOELibrary(BaseDriverLibrary[T, BaseVariableSpace], Serializable):
    """Base class for libraries of DOEs."""

    samples: RealArray
    """The samples in the input space.

    The variable types of the input space are stored as dtype metadata.

    To access those in the unit hypercube, use
    [unit_samples][gemseo.doe.core.base_doe_library.BaseDOELibrary.unit_samples].
    """

    unit_samples: RealArray
    """The samples projected in the unit hypercube.

    In the case of an input space of dimension $d$,
    the unit hypercube is $[0,1]^d$.

    To access those in the input space,
    use [samples][gemseo.doe.core.base_doe_library.BaseDOELibrary.samples].
    """

    _lock: RLock
    """The lock protecting database storage in multiprocessing."""

    _seeder: Seeder
    """A seed generator."""

    _use_unit_hypercube: ClassVar[bool] = True
    """Whether the algorithms use a unit hypercube to generate the samples."""

    _result_class: ClassVar[type[OptimizationResult]] = OptimizationResult
    """The class used to present the result when solving an optimization problem."""

    __output_functions: list[ArrayFunction] | None
    """The functions to compute the outputs, if any."""

    __jacobian_functions: list[ArrayFunction] | None
    """The functions to compute the Jacobians, if any."""

    __UNIT_HYPERCUBE_VARIABLE: Final[str] = "__unit_hypercube"
    """The name of the variable spanning the unit hypercube sampled by dimension.

    This name is not a valid name for a variable of a caller,
    so that a setting referring to a variable name cannot match it.
    """

    _attr_not_to_serialize: ClassVar[set[str]] = {"_lock"}

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self.samples = array([])
        self.unit_samples = array([])
        self._seeder = Seeder()
        self.__output_functions = []
        self.__jacobian_functions = []
        self._lock = RLock()

    def _init_shared_memory_attrs_after(self) -> None:
        self._lock = RLock()

    @property
    def seed(self) -> int:
        """The default seed used for reproducibility reasons."""
        return self._seeder.default_seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seeder.default_seed = value

    @property
    def _is_solving_optimization_problem(self) -> bool:
        """Whether is solving an optimization problem."""
        return super()._is_solving_optimization_problem and self._settings.eval_func

    def _pre_run(
        self,
        problem: EvaluationProblem,
    ) -> None:
        super()._pre_run(problem)
        problem.stop_if_nan = False

        input_space = problem.input_space
        # The space configures itself for its own mapping from the unit hypercube,
        # as in sample_space().
        with input_space._prepare_untransformation(self._use_unit_hypercube):
            self.unit_samples = self._generate_unit_samples(input_space)
            logger.debug(
                (
                    "The DOE algorithm %s of %s has generated %s samples "
                    "in the input unit hypercube of dimension %s."
                ),
                self._algo_name,
                self.__class__.__name__,
                *self.unit_samples.shape,
            )
            self.samples = self.__convert_unit_samples_to_samples(problem)

        self._init_iter_observer(
            problem,
            len(self.unit_samples),
            progress_bar_data_name=self._settings.progress_bar_data_name,
        )

    def __convert_unit_samples_to_samples(
        self, problem: EvaluationProblem
    ) -> RealArray:
        """Convert the unit samples to samples of the input space.

        We also set the variable types as dtype metadata.

        Args:
            problem: The problem to be solved.

        Returns:
            The samples.
        """
        input_space = problem.input_space
        samples = input_space.untransform_vect(self.unit_samples, no_check=True)
        variable_types = {
            name: variable.type for name, variable in input_space.variables.items()
        }
        unique_variable_types = set(variable_types.values())
        if len(unique_variable_types) > 1:
            # When the input space has both real and integer variables,
            # the samples array has the float dtype.
            # We record the integer variables types to later be able to restore the
            # proper data type.
            python_var_types = {
                name: data_type_to_numpy_type[type_]
                for name, type_ in variable_types.items()
                if type_ != DesignSpace.DesignVariableType.REAL
            }
            samples.dtype = dtype(samples.dtype, metadata=python_var_types)

        return samples

    @abstractmethod
    def _generate_unit_samples(self, input_space: BaseVariableSpace) -> RealArray:
        """Generate the samples of the input value in the unit hypercube.

        Args:
            input_space: The input space to be sampled.

        Returns:
            The samples of the input value in the unit hypercube.
        """

    def _run(self, problem: EvaluationProblem) -> None:
        output_functions, jacobian_functions = self._problem.get_functions(
            jacobian_names=() if self._settings.eval_jac else None, observable_names=()
        )
        self.__output_functions = (
            output_functions if self._settings.eval_func and output_functions else None
        )
        self.__jacobian_functions = jacobian_functions or None
        if self._settings.n_processes > 1:
            self.__run_in_parallel_one_at_a_time()
        elif self._settings.vectorize:
            self.__run_in_serial_all_at_once()
        else:
            self.__run_in_serial_one_at_a_time()

    def __run_in_serial_one_at_a_time(self) -> None:
        """Evaluate the functions in serial, sample by sample."""
        for index, input_value in enumerate(self.samples):
            for preprocessor in self._settings.preprocessors:
                preprocessor(index)

            try:
                result = self._evaluate_functions(input_value, sample_index=index)
            except ValueError:  # noqa: PERF203
                logger.exception(
                    "The evaluation of the functions at point %s raised a"
                    " ValueError; skipping to the next point.",
                    input_value,
                )
                self._problem.evaluation_counter.enabled = False
                continue

            for callback in self._settings.callbacks:
                callback(index, result)

            if not self._settings.use_database:
                self._problem.evaluation_counter.current += 1
                if self._progress_bar is not None:
                    self._progress_bar.update(HashableNdarray(input_value))

    def __run_in_serial_all_at_once(self) -> None:
        """Evaluate the functions in serial, with all the samples at once."""
        output_values, jacobian_values = self._evaluate_functions(self.samples)
        n_samples = len(self.samples)
        jacobian_shapes = {
            output_name: (
                value.shape[0] // n_samples,
                value.shape[1] // n_samples,
            )
            for output_name, value in jacobian_values.items()
        }
        output_shape = (n_samples, -1)
        for callback in self._settings.callbacks:
            output_values = {
                name: value.reshape(output_shape)
                for name, value in output_values.items()
            }
            for index in range(len(self.samples)):
                callback(
                    index,
                    (
                        {name: value[index] for name, value in output_values.items()},
                        {
                            name: jacobian_values[name][
                                index * d : (index + 1) * d,
                                index * p : (index + 1) * p,
                            ]
                            for name, (d, p) in jacobian_shapes.items()
                        },
                    ),
                )

    def __run_in_parallel_one_at_a_time(self) -> None:
        """Evaluate the functions in parallel, sample by sample."""
        logger.info(
            "Running DOE in parallel on n_processes = %s", self._settings.n_processes
        )
        # Given a ndarray input value,
        # the worker evaluates the functions attached to the problem
        # with up to n_processes simultaneous processes.
        parallel = CallableParallelExecution(
            [self._worker],
            n_processes=self._settings.n_processes,
            wait_time_between_fork=self._settings.wait_time_between_samples,
        )
        database = self._problem.database
        callbacks = list(self._settings.callbacks)

        # Add a callback
        # to store the samples in the database on the fly (when use_database=True)
        # and update the progress bar.
        callbacks.append(self.__store_in_database_and_finalize_iteration)

        if self._settings.use_database:
            # Initialize the order of samples in the database
            # as parallel execution does not guarantee it.
            for sample in self.samples:
                database.store(sample, {})

        # The list of inputs of the tasks is the list of samples with their
        # indices, so that the workers know the sample they evaluate.
        # A callback function stores the samples on the fly
        # during the parallel execution.
        self._problem.evaluation_counter.enabled = True
        parallel.execute(
            list(enumerate(self.samples)),
            exec_callbacks=callbacks,
            preprocessors=self._settings.preprocessors,
        )
        if self._settings.use_database:
            # We added empty entries by default to keep order in the database
            # but when the DOE point is failed, this is not consistent
            # with the serial exec, so we clean the DB
            database.remove_empty_entries()

    def _worker(self, indexed_input_value: tuple[int, RealArray]) -> EvaluationType:
        """Evaluate the functions at a given input point.

        To be used by
        [CallableParallelExecution][gemseo.core.parallel_execution.callable_parallel_execution.CallableParallelExecution].

        Args:
            indexed_input_value: The index of the sample and the input point.

        Returns:
            The output value and the Jacobian value.
        """
        if parent_process() is not None:
            self._progress_bar = None
            self._problem.database.clear_listeners()

        sample_index, input_value = indexed_input_value
        return self._evaluate_functions(input_value, sample_index=sample_index)

    def _evaluate_functions(
        self,
        input_value: RealArray,
        sample_index: int = -1,
    ) -> EvaluationType:
        """Evaluate the functions at a given input point.

        Args:
            input_value: The input point.
            sample_index: The index of the input point in the samples.
                A negative value means that the index is unknown or does not
                apply (e.g. all the samples are evaluated at once).
                It is read by the workflow observers to name the directory of
                the sample independently of the evaluation order.

        Returns:
            The output value and the Jacobian value.
        """
        return self._problem.evaluate_functions(
            input_value=input_value,
            preprocess_input_value=False,
            input_value_is_normalized=False,
            output_functions=self.__output_functions,
            jacobian_functions=self.__jacobian_functions,
        )

    @synchronized
    def __store_in_database_and_finalize_iteration(
        self,
        index: int,
        output_and_jacobian_data: EvaluationType,
    ) -> None:
        """Update the progress bar.

        Args:
            index: The sample index.
            output_and_jacobian_data: The output and Jacobian data.
        """
        if self._settings.use_database:
            data, jacobian_data = output_and_jacobian_data
            if jacobian_data:
                for output_name, jacobian in jacobian_data.items():
                    data[self._problem.database.get_gradient_name(output_name)] = (
                        jacobian
                    )

            input_value = self.samples[index]
            self._problem.database.store(input_value, data)
            input_value = HashableNdarray(input_value)
        else:
            input_value = None

        self._problem.evaluation_counter.current += 1
        if self._progress_bar is not None:
            self._progress_bar.update(input_value)

    def sample_space(
        self,
        space: BaseVariableSpace | int,
        settings: BaseDOESettings | None = None,
        use_unit_samples: bool = False,
    ) -> RealArray:
        """Sample a variable space.

        The space is sampled through its mapping from the unit hypercube,
        which is geometric for a
        [DesignSpace][gemseo.space.design.DesignSpace]
        and iso-probabilistic for a
        [RandomSpace][gemseo.space.random.RandomSpace].

        Args:
            space: The variables space.
            settings: The settings of the DOE algorithm.
                If `None`, use the default settings.
            use_unit_samples: Whether to return unit samples,
                i.e. samples distributed in the unit hypercube.

        Returns:
            The design of experiments
            whose rows are the samples and columns the variables.
        """
        self._settings = create_model(
            self.ALGORITHM_INFOS[self.algo_name].settings_class,
            settings_model=settings,
        )
        if use_unit_samples:
            return self._generate_unit_samples(space)

        # The space configures itself for its own mapping from the unit hypercube.
        with space._prepare_untransformation(self._use_unit_hypercube):
            unit_samples = self._generate_unit_samples(space)
            return space.untransform_vect(unit_samples, no_check=True)

    def sample_unit_hypercube(
        self,
        dimension: int,
        settings: BaseDOESettings | None = None,
    ) -> RealArray:
        r"""Sample the unit hypercube $[0,1]^d$.

        Args:
            dimension: The dimension $d$ of the hypercube.
            settings: The settings of the DOE algorithm.
                If `None`, use the default settings.

        Returns:
            The design of experiments
            whose rows are the samples and columns the variables.

        Note:
            The hypercube has no variable name,
            so a setting referring to one has no effect here,
            e.g. the `reverse` setting of
            [DiagonalDOE][gemseo.doe.diagonal_doe.diagonal_doe.DiagonalDOE]
            given variable names rather than component indices.
        """
        # The algorithms sample a space rather than a dimension,
        # so carry the dimension in a space whose single variable spans it.
        # Its name cannot be the name of a variable of the caller,
        # otherwise a setting referring to that name would match this variable.
        space = DesignSpace()
        space.add_variable(
            self.__UNIT_HYPERCUBE_VARIABLE,
            size=dimension,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        self._settings = create_model(
            self.ALGORITHM_INFOS[self.algo_name].settings_class,
            settings_model=settings,
        )
        return self._generate_unit_samples(space)
