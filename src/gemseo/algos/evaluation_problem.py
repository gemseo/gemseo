# Copyright 2022 Airbus SAS
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
#       :author: Damien Guenot
#       :author: Francois Gallard, Charlie Vanaret, Benoit Pauwels
#       :author: Gabriel Max De Mendonça Abrantes
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Evaluation problem."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Literal
from typing import Union
from typing import overload

from numpy import any as np_any
from strenum import StrEnum

from gemseo.algos.base_problem import BaseProblem
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.evaluation_counter import EvaluationCounter
from gemseo.algos.problem_function import ProblemFunction
from gemseo.core.mdo_functions.collections.observables import Observables
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
from gemseo.datasets.dataset import Dataset
from gemseo.datasets.io_dataset import IODataset
from gemseo.typing import RealArray
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.enumeration import merge_enums
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.core.mdo_functions.mdo_function import MDOFunction


LOGGER = logging.getLogger(__name__)

EvaluationType = tuple[dict[str, Union[float, RealArray]], dict[str, RealArray]]
"""The type of the output value of an evaluation."""


class EvaluationProblem(BaseProblem):
    """A problem to evaluate functions over a design space.

    This problem can only include observables,
    i.e. functions with :attr:`~.MDOFunction.FunctionType.OBS` as function type.
    """

    __new_iter_observables: Observables
    """The observables to be evaluated whenever a database entry is created."""

    __observables: Observables
    """The observables."""

    check_bounds: ClassVar[bool] = True
    """Whether to check if a point is in the design space before calling functions."""

    database: Database
    """The database to store the function evaluations."""

    design_space: DesignSpace
    """The design space on which the functions are evaluated."""

    evaluation_counter: EvaluationCounter
    """The counter of function evaluations.

    Every execution of a :class:`.DriverLibrary` handling this problem
    increments this counter by 1.
    """

    differentiation_step: float
    """The differentiation step."""

    _stop_if_nan: bool
    """Whether the evaluation stops when a function returns ``NaN``."""

    ApproximationMode = ApproximationMode

    class _DifferentiationMethod(StrEnum):
        """The additional differentiation methods."""

        USER_GRAD = "user"
        NO_DERIVATIVE = "no_derivative"

    DifferentiationMethod = merge_enums(
        "DifferentiationMethod",
        StrEnum,
        ApproximationMode,
        _DifferentiationMethod,
        doc="The differentiation methods.",
    )

    differentiation_method: DifferentiationMethod
    """The differentiation method."""

    def __init__(
        self,
        design_space: DesignSpace,
        database: Database | None = None,
        differentiation_method: DifferentiationMethod = DifferentiationMethod.USER_GRAD,
        differentiation_step: float = 1e-7,
        parallel_differentiation: bool = False,
        **parallel_differentiation_options: int | bool,
    ) -> None:
        """
        Args:
            design_space: The design space on which the functions are evaluated.
            database: The initial database to store the function evaluations.
                If ``None``,
                the problem starts from an empty database.
                If there is no need to store the function evaluations,
                this argument is ignored.
            differentiation_method: The differentiation method
                to evaluate the derivatives.
            differentiation_step: The step used by the differentiation method.
                This argument is ignored
                when the differentiation method is not an :attr:`.ApproximationMode`.
            parallel_differentiation: Whether
                to approximate the derivatives in parallel.
            hdf_node_path: The path of the node in the HDF file
                to store the function evaluations.
                If empty, the root node is considered.
                This argument is ignored
                when ``database`` is a ``Database`` or the empty string.
            **parallel_differentiation_options: The options
                to approximate the derivatives in parallel.
        """  # noqa: D205, D212, D415
        self._functions_are_preprocessed = False
        self.__observables = Observables()
        self.__new_iter_observables = Observables()
        self.differentiation_step = differentiation_step
        self.differentiation_method = differentiation_method
        self.database = (
            Database(input_space=design_space) if database is None else database
        )
        self.design_space = design_space
        self.__initial_current_x = deepcopy(
            design_space.get_current_value(as_dict=True)
        )
        self._stop_if_nan = True
        self.__parallel_differentiation = parallel_differentiation
        self.__parallel_differentiation_options = parallel_differentiation_options
        self.evaluation_counter = EvaluationCounter()
        self._sequence_of_functions = [self.__observables, self.__new_iter_observables]
        self._function_names = []

    def __repr__(self) -> str:
        return str(self._get_string_representation())

    def _repr_html_(self) -> str:
        return self._get_string_representation()._repr_html_()

    def _get_string_representation(self) -> MultiLineString:
        """Return the string representation of the evaluation problem.

        Returns:
            The string representation of the evaluation problem.
        """
        mls = MultiLineString()
        mls.add("Evaluation problem:")
        mls.indent()
        mls.add("Evaluate the functions: {}", pretty_str(self.function_names))
        return mls

    @property
    def stop_if_nan(self) -> bool:
        """Whether the evaluation stops when a function returns ``NaN``."""
        return self._stop_if_nan

    @stop_if_nan.setter
    def stop_if_nan(self, value: bool) -> None:
        self._stop_if_nan = value
        for functions in self._sequence_of_functions:
            for function in functions:
                if isinstance(function, ProblemFunction):
                    function.stop_if_nan = value

        for function_name in self._function_names:
            function = getattr(self, function_name)
            if isinstance(function, ProblemFunction):
                function.stop_if_nan = value

    def __check_functions_are_not_preprocessed(self) -> None:
        """Raise an exception if the functions have already been pre-processed.

        Raises:
            RuntimeError: When the functions have already been pre-processed.
        """
        if self._functions_are_preprocessed:
            msg = (
                "The parallel differentiation cannot be changed "
                "because the functions have already been pre-processed."
            )
            raise RuntimeError(msg)

    @property
    def parallel_differentiation(self) -> bool:
        """Whether to approximate the derivatives in parallel.

        This attribute is ignored
        when the differentiation method is not an :attr:`.ApproximationMode`.
        """
        return self.__parallel_differentiation

    @parallel_differentiation.setter
    def parallel_differentiation(self, value: bool) -> None:
        self.__check_functions_are_not_preprocessed()
        self.__parallel_differentiation = value

    @property
    def parallel_differentiation_options(self) -> dict[str, int | bool]:
        """The options to approximate the derivatives in parallel.

        This attribute is ignored
        when the differentiation method is not an :attr:`.ApproximationMode`.
        """
        return self.__parallel_differentiation_options

    @parallel_differentiation_options.setter
    def parallel_differentiation_options(self, value: dict[str, int | bool]) -> None:
        self.__check_functions_are_not_preprocessed()
        self.__parallel_differentiation_options = value

    @property
    def observables(self) -> Observables:
        """The observables."""
        return self.__observables

    @observables.setter
    def observables(self, functions: Iterable[MDOFunction]) -> None:
        self.__observables.clear()
        self.__observables.extend(functions)

    @property
    def new_iter_observables(self) -> Observables:
        """The observables to be evaluated whenever a database entry is created."""
        return self.__new_iter_observables

    @new_iter_observables.setter
    def new_iter_observables(self, functions: Iterable[MDOFunction]) -> None:
        self.__new_iter_observables.clear()
        self.__new_iter_observables.extend(functions)

    def add_observable(
        self,
        observable: MDOFunction,
        new_iter: bool = True,
    ) -> None:
        """Add an observable function.

        It is an :class:`.MDOFunction`
        with :attr:`~.MDOFunction.FunctionType.OBS` as function type.

        Args:
            observable: The observable function.
            new_iter: Whether to call the observable
                whenever a database entry is created.
        """
        formatted_observable = self.__observables.format(observable)
        if formatted_observable is None:
            return

        self.__observables.append(formatted_observable)
        if new_iter:
            self.__new_iter_observables.append(formatted_observable)

    @property
    def functions(self) -> list[MDOFunction]:
        """All the functions except :attr:`.new_iter_observables`."""
        return list(self.__observables)

    @property
    def original_functions(self) -> list[MDOFunction]:
        """All the original functions except those of :attr:`.new_iter_observables`."""
        return list(self.__observables.get_originals())

    @property
    def function_names(self) -> list[str]:
        """All the function names except those of :attr:`.new_iter_observables`."""
        return [function.name for function in self.functions]

    def add_listener(
        self,
        listener: Callable[[RealArray], Any],
        at_each_iteration: bool = True,
        at_each_function_call: bool = False,
    ) -> None:
        """Add a listener for some events.

        Listeners are callback functions attached to the database
        which are triggered when new values are stored within the database.

        Args:
            listener: A function to be called after some events,
                whose argument is a design vector.
            at_each_iteration: Whether to evaluate the listeners
                after evaluating all functions
                for a given point and storing their values in the :attr:`.database`.
            at_each_function_call: Whether to evaluate the listeners
                after storing any new value in the :attr:`.database`.
        """
        if at_each_function_call:
            self.database.add_store_listener(listener)
        if at_each_iteration:
            self.database.add_new_iter_listener(listener)

    def get_functions(
        self,
        no_db_no_norm: bool = False,
        observable_names: Iterable[str] | None = None,
        jacobian_names: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[MDOFunction], list[MDOFunction]]:
        """Return the functions to be evaluated.

        Args:
            no_db_no_norm: Whether to prevent
                both database backup and design vector normalization.
            observable_names: The names of the observables to evaluate.
                If empty,
                then all the observables are evaluated.
                If ``None``,
                then no observable is evaluated.
            jacobian_names: The names of the functions
                whose Jacobian matrices must be computed.
                If empty,
                then compute the Jacobian matrices of the functions
                that are selected for evaluation using the other arguments.
                If ``None``,
                then no Jacobian matrices is computed.
            **kwargs: The options to select the functions to be evaluated.

        Returns:
            The functions computing the outputs
            and the functions computing the Jacobians.

        Raises:
            ValueError: If a name in ``jacobian_names`` is not the name of
                a function of the problem.
        """
        output_functions = self._get_functions(
            observable_names, no_db_no_norm, **kwargs
        )
        if jacobian_names is None:
            return output_functions, []

        if not jacobian_names:
            return output_functions, output_functions

        unknown_names = set(jacobian_names) - set(self.function_names)
        if unknown_names:
            message = "These names are" if len(unknown_names) > 1 else "This name is"

            msg = (
                f"{message} not among the names of the functions: "
                f"{pretty_str(unknown_names)}."
            )
            raise ValueError(msg)

        observable_names = [
            name for name in jacobian_names if name in self.__observables.get_names()
        ]
        jacobian_functions = self._get_functions(
            observable_names or None,
            no_db_no_norm,
            **self._get_options_for_get_functions(jacobian_names),
        )
        return output_functions, jacobian_functions

    def evaluate_functions(
        self,
        design_vector: RealArray | None = None,
        design_vector_is_normalized: bool = True,
        preprocess_design_vector: bool = True,
        output_functions: Iterable[MDOFunction] | None = (),
        jacobian_functions: Iterable[MDOFunction] | None = None,
    ) -> EvaluationType:
        """Evaluate the functions, and possibly their derivatives.

        Args:
            design_vector: The design vector at which to evaluate the functions;
                if ``None``, use the current value of the design space.
            design_vector_is_normalized: Whether ``design_vector`` is normalized.
            preprocess_design_vector: Whether to preprocess the design vector.
            output_functions: The functions computing the outputs.
                If empty, evaluate all the functions computing outputs.
                If ``None``, do not evaluate functions computing outputs.
            jacobian_functions: The functions computing the Jacobians.
                If empty, evaluate all the functions computing Jacobians.
                If ``None``, do not evaluate functions computing Jacobians.

        Returns:
            The output values of the functions,
            as well as their Jacobian matrices if ``jacobian_functions`` is empty.
        """
        if output_functions is None and jacobian_functions is None:
            return {}, {}

        use_all_output_functions = not output_functions and output_functions is not None
        use_all_jacobian_functions = (
            not jacobian_functions and jacobian_functions is not None
        )
        if use_all_output_functions or use_all_jacobian_functions:
            all_output_functions, all_jacobian_functions = self.get_functions(
                jacobian_names=()
            )
            if use_all_output_functions:
                output_functions = all_output_functions

            if use_all_jacobian_functions:
                jacobian_functions = all_jacobian_functions

        if output_functions is None:
            output_functions = ()

        if jacobian_functions is None:
            jacobian_functions = ()

        if preprocess_design_vector:
            functions = output_functions or jacobian_functions
            if functions:
                # N.B. either all functions expect normalized inputs or none of them do.
                design_vector = self._preprocess_inputs(
                    design_vector,
                    design_vector_is_normalized,
                    functions[0].expects_normalized_inputs,
                )

        outputs = {}
        for function in output_functions:
            try:
                outputs[function.name] = function.evaluate(design_vector)
            except ValueError:  # noqa: PERF203
                LOGGER.exception("Failed to evaluate function %s", function.name)
                raise

        if not jacobian_functions:
            return outputs, {}

        jacobians = {}
        for function in jacobian_functions:
            try:
                jacobians[function.name] = function.jac(design_vector)
            except ValueError:  # noqa: PERF203
                LOGGER.exception("Failed to evaluate Jacobian of %s.", function.name)
                raise

        return outputs, jacobians

    def _get_options_for_get_functions(
        self, jacobian_names: list[str]
    ) -> dict[str, Any]:
        """Return the options for :meth:`._get_functions.

        Args:
            jacobian_names: The names of the functions
                whose Jacobian matrices must be computed.

        Returns:
            The options for :meth:`._get_functions.
        """
        return {}

    def _preprocess_inputs(
        self,
        input_value: RealArray | None,
        normalized: bool,
        normalization_expected: bool,
    ) -> RealArray:
        """Prepare the design variables for the function evaluation.

        Args:
            input_value: The design variables.
                If ``None``, use the current value of the design space.
            normalized: Whether the design variables are normalized.
            normalization_expected: Whether the functions expect normalized variables.

        Returns:
            The prepared design variables.
        """
        if input_value is None:
            input_value = self.design_space.get_current_value(normalize=normalized)
        elif self.check_bounds:
            if normalized:
                non_normalized_variables = self.design_space.unnormalize_vect(
                    input_value, no_check=True
                )
            else:
                non_normalized_variables = input_value

            self.design_space.check_membership(non_normalized_variables)

        if normalized and not normalization_expected:
            return self.design_space.unnormalize_vect(input_value, no_check=True)

        if not normalized and normalization_expected:
            return self.design_space.normalize_vect(input_value)

        return input_value

    def _get_functions(
        self,
        observable_names: Iterable[str] | None,
        no_db_no_norm: bool,
        **kwargs: Any,
    ) -> list[MDOFunction]:
        """Return functions.

        Args:
            observable_names: The names of the observables to return.
                If empty,
                then all the observables are returned.
                If ``None``,
                then no observable is returned.
            no_db_no_norm: Whether to prevent
                both database backup and design vector normalization.
            *kwargs: The options to select the functions to be evaluated.

        Returns:
            The functions.
        """
        from_original_functions = not self._functions_are_preprocessed or no_db_no_norm
        functions = []
        if observable_names is None:
            return functions

        if observable_names:
            return [
                self.observables.get_from_name(name, from_original_functions)
                for name in observable_names
            ]

        if from_original_functions:
            return list(self.__observables.get_originals())

        return list(self.__observables)

    def preprocess_functions(
        self,
        is_function_input_normalized: bool = True,
        use_database: bool = True,
        round_ints: bool = True,
        eval_obs_jac: bool = False,
        support_sparse_jacobian: bool = False,
        store_jacobian: bool = True,
    ) -> None:
        """Wrap the function for a more attractive evaluation.

        E.g. approximation of the derivatives or evaluation backup.

        In the case of derivative approximation,
        only the computed gradients are stored in the database,
        not the eventual finite differences or complex step perturbed evaluations.

        Args:
            is_function_input_normalized: Whether to consider the function input as
                normalized and unnormalize it before the function evaluation.
            use_database: Whether to store the function evaluations in the database.
            round_ints: Whether to round the integer variables.
            eval_obs_jac: Whether to evaluate the Jacobian of the observables.
            support_sparse_jacobian: Whether the driver supports sparse Jacobian.
            store_jacobian: Whether to store the Jacobian matrices in the database.
                This argument is ignored when ``use_database`` is ``False``.
        """
        # Avoids multiple wrappings of functions when multiple executions
        # are performed, in bi-level scenarios for instance
        if self._functions_are_preprocessed:
            return

        if round_ints:
            # Keep the rounding option only if there is an integer design variable
            round_ints = any(
                np_any(variable_type == DesignSpace.DesignVariableType.INTEGER)
                for variable_type in self.design_space.variable_types.values()
            )

        for functions in self._sequence_of_functions:
            if functions == self.__new_iter_observables:
                is_function_input_normalized_ = False
            else:
                is_function_input_normalized_ = is_function_input_normalized
            for index, function in enumerate(functions):
                function = self._preprocess_function(
                    function,
                    is_function_input_normalized=is_function_input_normalized_,
                    use_database=use_database,
                    round_ints=round_ints,
                    support_sparse_jacobian=support_sparse_jacobian,
                    store_jacobian=store_jacobian,
                )
                functions[index] = function

        for function_name in self._function_names:
            setattr(
                self,
                function_name,
                self._preprocess_function(
                    getattr(self, function_name),
                    is_function_input_normalized=is_function_input_normalized,
                    use_database=use_database,
                    round_ints=round_ints,
                    support_sparse_jacobian=support_sparse_jacobian,
                    store_jacobian=store_jacobian,
                ),
            )
        self._functions_are_preprocessed = True
        self.check()
        self.new_iter_observables.evaluate_jacobian = eval_obs_jac

    @staticmethod
    def _convert_array_to_dense(value):
        return value.todense() if isinstance(value, sparse_classes) else value

    def _preprocess_function(
        self,
        function: MDOFunction,
        is_function_input_normalized: bool = True,
        use_database: bool = True,
        round_ints: bool = True,
        support_sparse_jacobian: bool = False,
        store_jacobian: bool = True,
    ) -> ProblemFunction:
        """Wrap the function for a more attractive evaluation.

        Args:
            function: The scaled and derived function to be pre-processed.
            is_function_input_normalized: Whether to consider the function input as
                normalized and unnormalize it before the function evaluation.
            use_database: Whether to store the function evaluations in the database.
            round_ints: Whether to round the integer variables.
            support_sparse_jacobian: Whether the driver supports sparse Jacobian.
            store_jacobian: Whether to store the Jacobian in the database.

        Returns:
            The pre-processed function.
        """
        original_function = function
        args = () if support_sparse_jacobian else (self._convert_array_to_dense,)
        ds = self.design_space
        if (
            isinstance(function, MDOLinearFunction)
            and not round_ints
            and is_function_input_normalized
        ):
            expects_normalized_inputs = True
            function = function.normalize(self.design_space)
            func_seq = (function.func,)
            jac_seq = (function.jac, *args)
        elif is_function_input_normalized and round_ints:
            expects_normalized_inputs = True
            func_seq = (ds.unnormalize_vect, ds.round_vect, function.func)
            jac_seq = (
                ds.unnormalize_vect,
                ds.round_vect,
                function.jac,
                *args,
                ds.normalize_grad,
            )
        elif round_ints:
            expects_normalized_inputs = function.expects_normalized_inputs
            func_seq = (ds.round_vect, function.func)
            jac_seq = (ds.round_vect, function.jac, *args)
        elif is_function_input_normalized:
            expects_normalized_inputs = True
            func_seq = (ds.unnormalize_vect, function.func)
            jac_seq = (ds.unnormalize_vect, function.jac, *args, ds.normalize_grad)
        else:
            expects_normalized_inputs = function.expects_normalized_inputs
            func_seq = (function.func,)
            jac_seq = (function.jac, *args)

        function = ProblemFunction(
            function,
            func_seq,
            jac_seq,
            expects_normalized_inputs,
            self.database if use_database else None,
            self.evaluation_counter,
            self.stop_if_nan,
            self.design_space,
            store_jacobian,
            differentiation_method=(
                None
                if (
                    self.differentiation_method
                    in set(self.DifferentiationMethod).difference(
                        set(self.ApproximationMode)
                    )
                )
                else self.differentiation_method
            ),
            step=self.differentiation_step,
            normalize=is_function_input_normalized,
            parallel=self.__parallel_differentiation,
            **self.__parallel_differentiation_options,
        )
        function.original = original_function
        return function

    def check(self) -> None:
        """Check if the functions attached to the problem can be evaluated."""
        self.design_space.check()

    @overload
    def to_dataset(
        self,
        name: str = ...,
        categorize: Literal[True] = ...,
        export_gradients: bool = ...,
        input_values: Iterable[RealArray] = ...,
        **dataset_options: ...,
    ) -> IODataset: ...

    @overload
    def to_dataset(
        self,
        name: str = ...,
        categorize: Literal[False] = ...,
        export_gradients: bool = ...,
        input_values: Iterable[RealArray] = ...,
        **dataset_options: ...,
    ) -> Dataset: ...

    def to_dataset(
        self,
        name: str = "",
        categorize: bool = True,
        export_gradients: bool = False,
        input_values: Iterable[RealArray] = (),
        **dataset_options: Any,
    ) -> Dataset:
        """Export the database of the problem to a :class:`.Dataset`.

        Args:
            name: The name to be given to the dataset.
                If empty,
                use the name of the :attr:`~.database`.
            categorize: Whether to distinguish
                between the different groups of variables.
                If so,
                use an :class:`.IODataset`
                with the design variables in the :attr:`.IODataset.INPUT_GROUP`
                and the functions and their derivatives
                in the :attr:`.IODataset.OUTPUT_GROUP`.
                Otherwise,
                group all the variables in :attr:`.Dataset.PARAMETER_GROUP``.
            export_gradients: Whether to export the gradients of the functions
                if the latter are available in the database of the problem.
            input_values: The input values to be considered.
                If empty, consider all the input values of the database.

        Returns:
            A dataset built from the database of the problem.
        """
        if categorize:
            dataset_class = IODataset
            input_group = IODataset.INPUT_GROUP
            output_group = IODataset.OUTPUT_GROUP
            gradient_group = Dataset.GRADIENT_GROUP
        else:
            dataset_class = Dataset
            input_group = output_group = gradient_group = Dataset.DEFAULT_GROUP

        return self.database.to_dataset(
            name=name,
            export_gradients=export_gradients,
            input_values=input_values,
            dataset_class=dataset_class,
            input_group=input_group,
            output_group=output_group,
            gradient_group=gradient_group,
        )

    def reset(
        self,
        database: bool = True,
        current_iter: bool = True,
        design_space: bool = True,
        function_calls: bool = True,
        preprocessing: bool = True,
    ) -> None:
        """Partially or fully reset the problem.

        Args:
            database: Whether to clear the database.
            current_iter: Whether to reset the counter of evaluations
                to the initial iteration.
            design_space: Whether to reset the current value of the design space
                which can be ``None``.
            function_calls: Whether to reset the number of calls of the functions.
            preprocessing: Whether to turn the pre-processing of functions to False.
        """
        if current_iter:
            self.evaluation_counter.current = 0

        if database:
            self.database.clear()

        if design_space:
            self.design_space.set_current_value(self.__initial_current_x)

        if function_calls and ProblemFunction.enable_statistics:
            for function in self.functions:
                function.n_calls = 0

            if self._functions_are_preprocessed:
                for original_functions in self.original_functions:
                    original_functions.n_calls = 0

        if preprocessing and self._functions_are_preprocessed:
            n_o_calls = [o.n_calls for o in self.__observables]
            n_nio_calls = [o.n_calls for o in self.__new_iter_observables]
            self.__observables.reset()
            self.__new_iter_observables.reset()
            if not function_calls and ProblemFunction.enable_statistics:
                for o, n_calls in zip(self.__observables, n_o_calls):
                    o.n_calls = n_calls
                for nio, n_calls in zip(self.__new_iter_observables, n_nio_calls):
                    nio.n_calls = n_calls

            self._functions_are_preprocessed = False
