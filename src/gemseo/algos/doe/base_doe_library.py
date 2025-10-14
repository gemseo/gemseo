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
from functools import singledispatchmethod
from multiprocessing import RLock
from multiprocessing import parent_process
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import TypeVar

from numpy import array
from numpy import dtype
from numpy import hstack
from numpy import where

from gemseo.algos.base_driver_library import BaseDriverLibrary
from gemseo.algos.base_driver_library import DriverDescription
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.algos.evaluation_problem import EvaluationType
from gemseo.algos.hashable_ndarray import HashableNdarray
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.core.serializable import Serializable
from gemseo.utils.locks import synchronized
from gemseo.utils.pydantic import create_model
from gemseo.utils.seeder import Seeder
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from gemseo.algos.base_driver_library import DriverSettingType
    from gemseo.algos.evaluation_problem import EvaluationProblem
    from gemseo.core.mdo_functions.mdo_function import MDOFunction
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


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

    Settings: type[BaseDOESettings] = BaseDOESettings
    """The Pydantic model for the DOE library settings."""


class BaseDOELibrary(BaseDriverLibrary[T], Serializable):
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

    _seeder: Seeder
    """A seed generator."""

    _USE_UNIT_HYPERCUBE: ClassVar[bool] = True
    """Whether the algorithms use a unit hypercube to generate the design samples."""

    __output_functions: list[MDOFunction] | None
    """The functions to compute the outputs, if any."""

    __jacobian_functions: list[MDOFunction] | None
    """The functions to compute the Jacobians, if any."""

    _ATTR_NOT_TO_SERIALIZE: ClassVar[set[str]] = {"lock"}

    def __init__(self, algo_name: str) -> None:  # noqa:D107
        super().__init__(algo_name)
        self.samples = array([])
        self.unit_samples = array([])
        self._seeder = Seeder()
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

        design_space = problem.design_space
        integer_normalization_enabled = self.__enable_integer_variables_normalization(
            design_space
        )
        self.__check_unnormalization_capability(design_space)

        self.unit_samples = self._generate_unit_samples(design_space)
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
        self.__reset_integer_variables_normalization(
            design_space, integer_normalization_enabled
        )
        self._init_iter_observer(
            problem,
            len(self.unit_samples),
            progress_bar_data_name=self._settings.progress_bar_data_name,
        )

    def __convert_unit_samples_to_samples(
        self, problem: EvaluationProblem
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
        unique_variable_types = set(variable_types.values())
        if len(unique_variable_types) > 1:
            # When the design space have both float and integer variables,
            # the samples array has the float dtype.
            # We record the integer variables types to later be able to restore the
            # proper data type.
            python_var_types = {
                name: DesignSpace.VARIABLE_TYPES_TO_DTYPES[type_]
                for name, type_ in variable_types.items()
                if type_ != DesignSpace.DesignVariableType.FLOAT
            }
            samples.dtype = dtype(samples.dtype, metadata=python_var_types)

        return samples

    @abstractmethod
    def _generate_unit_samples(self, design_space: DesignSpace) -> RealArray:
        """Generate the samples of the design vector in the unit hypercube.

        Args:
            design_space: The design space to be sampled.

        Returns:
            The samples of the design vector in the unit hypercube.
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
                result = self._evaluate_functions(input_value)
            except ValueError:  # noqa: PERF203
                LOGGER.exception(
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
        LOGGER.info(
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

        # The list of inputs of the tasks is the list of samples
        # A callback function stores the samples on the fly
        # during the parallel execution.
        self._problem.evaluation_counter.enabled = True
        parallel.execute(
            self.samples,
            exec_callback=callbacks,
            preprocessors=self._settings.preprocessors,
        )
        if self._settings.use_database:
            # We added empty entries by default to keep order in the database
            # but when the DOE point is failed, this is not consistent
            # with the serial exec, so we clean the DB
            database.remove_empty_entries()

    def _worker(self, input_value: RealArray) -> EvaluationType:
        """Evaluate the functions at a given input point.

        To be used by :class:`.CallableParallelExecution`.

        Args:
            input_value: The input point.

        Returns:
            The output value and the Jacobian value.
        """
        if parent_process() is not None:
            self._progress_bar = None
            self._problem.database.clear_listeners()

        return self._evaluate_functions(input_value)

    def _evaluate_functions(self, input_value: RealArray) -> EvaluationType:
        """Evaluate the functions at a given input point.

        Args:
            input_value: The input point.

        Returns:
            The output value and the Jacobian value.
        """
        return self._problem.evaluate_functions(
            design_vector=input_value,
            preprocess_design_vector=False,
            design_vector_is_normalized=False,
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
            msg = (
                f"The components {pretty_str(components, use_and=True)} "
                "of the design space are unbounded."
            )
            raise ValueError(msg)

    def compute_doe(
        self,
        variables_space: DesignSpace | int,
        unit_sampling: bool = False,
        settings_model: BaseDOESettings | None = None,
        **settings: DriverSettingType,
    ) -> RealArray:
        """Compute a design of experiments (DOE) in a variables space.

        Args:
            variables_space: Either the variables space to be sampled or its dimension.
            unit_sampling: Whether to sample in the unit hypercube.
                If the value provided in ``variables_space`` is the dimension,
                the samples will be generated in the unit hypercube
                whatever the value of ``unit_sampling``.
            settings_model: The DOE settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **settings: The DOE settings.
                These arguments are ignored when ``settings_model`` is not ``None``.

        Returns:
            The design of experiments
            whose rows are the samples and columns the variables.
        """
        design_space = self.__get_design_space(variables_space)
        if not unit_sampling:
            if isinstance(design_space, DesignSpace):
                integer_normalization_enabled = (
                    self.__enable_integer_variables_normalization(design_space)
                )

            self.__check_unnormalization_capability(design_space)

        self._settings = create_model(
            self.ALGORITHM_INFOS[self.algo_name].Settings,
            settings_model=settings_model,
            **settings,
        )

        unit_samples = self._generate_unit_samples(design_space)
        if unit_sampling:
            return unit_samples

        samples = design_space.untransform_vect(unit_samples, no_check=True)
        if isinstance(design_space, DesignSpace):
            self.__reset_integer_variables_normalization(
                design_space, integer_normalization_enabled
            )

        return samples

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
        design_space_.add_variable(
            "x", size=design_space, lower_bound=0.0, upper_bound=1.0
        )
        return design_space_

    @staticmethod
    def __enable_integer_variables_normalization(design_space: DesignSpace) -> bool:
        """Enable the normalization of the integer variables, if disabled.

        Args:
            design_space: The design space.

        Returns:
            Whether the normalization of the integer variables had to be enabled.

        """
        enabled = not design_space.enable_integer_variables_normalization
        if enabled:
            design_space.enable_integer_variables_normalization = True

        return enabled

    @staticmethod
    def __reset_integer_variables_normalization(
        design_space: DesignSpace, enabled: bool
    ) -> None:
        """Reset the normalization of the integer variables to its initial state.

        Args:
            design_space: The design space.
            enabled: Whether the normalization of the integer variables
                had to be enabled.
        """
        if enabled:
            design_space.enable_integer_variables_normalization = False
