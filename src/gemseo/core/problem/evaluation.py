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
from copy import copy
from copy import deepcopy
from enum import StrEnum
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import Literal
from typing import TypeVar
from typing import cast
from typing import overload

from gemseo.core.function.base_problem_function import BaseProblemFunction
from gemseo.core.function.collection.observables import Observables
from gemseo.core.function.evaluation_function import EvaluationFunction
from gemseo.core.function.transformed_input_function import TransformedInputFunction
from gemseo.core.problem.base import BaseProblem
from gemseo.core.problem.counter import EvaluationCounter
from gemseo.core.problem.database import Database
from gemseo.dataset.dataset import Dataset
from gemseo.dataset.io_dataset import IODataset
from gemseo.util.constant import _check_desvars_bounds
from gemseo.util.constant import _enable_working_database
from gemseo.util.derivative.approximation_mode import ApproximationMode
from gemseo.util.string import MultiLineString
from gemseo.util.string import pretty_str
from gemseo.util.typing import RealArray

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping

    from numpy import ndarray

    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.space.base import BaseVariableSpace
    from gemseo.space.design import DesignSpace
    from gemseo.space.transformation.base import BaseSpaceTransformation
    from gemseo.util.typing import StrPath


logger = logging.getLogger(__name__)

EvaluationType = tuple[dict[str, float | RealArray], dict[str, RealArray]]
"""The type of the output value of an evaluation."""

_SpaceT = TypeVar("_SpaceT", bound="BaseVariableSpace")


class EvaluationProblem(BaseProblem, Generic[_SpaceT]):
    """A problem to evaluate functions over an input space.

    Add the functions with
    [add_observable][gemseo.core.problem.evaluation.EvaluationProblem.add_observable]
    and evaluate them at a point of the space with
    [evaluate_functions][gemseo.core.problem.evaluation.EvaluationProblem.evaluate_functions],
    with their Jacobians when asked to,
    either the ones the functions provide
    or an approximation settled by
    [differentiation_method][gemseo.core.problem.evaluation.EvaluationProblem.differentiation_method].
    Unless asked otherwise,
    every evaluation is recorded in the
    [database][gemseo.core.problem.evaluation.EvaluationProblem.database],
    keyed on the point in the coordinates of the input space,
    which spares a second evaluation at the same point
    and can be exported with
    [to_dataset][gemseo.core.problem.evaluation.EvaluationProblem.to_dataset]
    or
    [to_hdf][gemseo.core.problem.evaluation.EvaluationProblem.to_hdf].

    Recording and counting the evaluations is the work of a wrapper,
    installed on every function by
    [bind_functions][gemseo.core.problem.evaluation.EvaluationProblem.bind_functions],
    whose `use_database` argument turns the recording off.
    The coordinates of the input space are the **original** ones;
    a transformation, normalizing or relaxing them,
    gives the **working** ones,
    and evaluating the functions in the working coordinates
    is the work of a second wrapper,
    which
    [create_working_problem][gemseo.core.problem.evaluation.EvaluationProblem.create_working_problem]
    installs on the **working problem**,
    a **new** problem built over the working space,
    leaving this one as it is and sharing its database with it.
    The wrappers of the working problem are the very ones this problem holds,
    not copies,
    and `create_working_problem` tells each the
    [transformation][gemseo.space.transformation.base.BaseSpaceTransformation],
    so an approximated Jacobian perturbs in the working coordinates
    whichever problem it is asked of
    and is recorded in the coordinates the user declared.
    Solving or sampling a problem performs both steps on its own,
    so they are there for whoever evaluates the functions by hand,
    to record the evaluations or to evaluate in the working coordinates.

    The type of the input space is the type parameter of this class:
    a subclass requiring a specific kind of space,
    e.g. an optimization problem requiring a design space,
    passes it to its base
    so that [input_space][gemseo.core.problem.evaluation.EvaluationProblem.input_space]
    is typed accordingly.
    """  # noqa: E501

    class HistoryFileFormat(StrEnum):
        """The format of the history file."""

        HDF5 = "hdf5"
        GGOBI = "ggobi"

    check_bounds: ClassVar[bool] = _check_desvars_bounds
    """Whether to check if a point is in the input space before calling functions.

    Driven by the `check_desvars_bounds` option of the global configuration.
    """

    enable_working_database: ClassVar[bool] = _enable_working_database
    """Whether an evaluation is also recorded in the working coordinates.

    This is a debugging aid, off by default.
    It observes a run without taking part in it:
    the store it writes to is one of its own,
    so it fires no listener, moves no counter and changes no value.
    Driven by the `enable_working_database` option of the global configuration.
    """

    _is_optimization: ClassVar[bool] = False
    """Whether the problem is an optimization problem."""

    _database: Database
    """The database to store the function evaluations."""

    differentiation_method: DifferentiationMethod
    """The differentiation method."""

    differentiation_step: float
    """The differentiation step.

    Taken in the coordinates the algorithm works in:
    a driver normalizing the design space moves a normalized component by it,
    which moves the one the user declared by the step times its range;
    a driver working in the coordinates the user declared
    moves the component itself.
    """

    evaluation_counter: EvaluationCounter
    """The counter of function evaluations.

    Every execution
    of a
    [BaseDriverLibrary][gemseo.core.algorithm.base_driver_library.BaseDriverLibrary]
    handling this problem increments this counter by 1.
    """

    __initial_current_value: Mapping[str, ndarray | None]
    """The initial current value of the input space.

    The current value is restored when the problem is reset;
    it is empty for a space defining none, e.g. a random space,
    and writing it back is then a no-op.
    """

    _input_space: _SpaceT
    """The input space on which the functions are evaluated."""

    __new_iter_observables: Observables
    """The observables to be evaluated whenever a database entry is created."""

    __observables: Observables
    """The observables."""

    _stop_if_nan: bool
    """Whether the evaluation stops when a function returns `NaN`."""

    working_database: Database | None = None
    """The evaluations in the working coordinates, when they are recorded.

    `None` until a driver builds the working problem with
    [create_working_problem][gemseo.core.problem.evaluation.EvaluationProblem.create_working_problem] while
    [enable_working_database][gemseo.core.problem.evaluation.EvaluationProblem.enable_working_database]
    is set,
    and so it stays under a driver building no working problem.
    The next run replaces it.
    This is a debugging aid: nothing of GEMSEO reads it.
    """  # noqa: E501

    ApproximationMode = ApproximationMode
    """The enumeration of approximation modes."""

    class DifferentiationMethod(StrEnum):
        """All differentiation methods, merging user/no-derivative and approximations.

        Each member aliases the corresponding sub-enum member so values stay in sync.
        """

        USER = "user"
        """User-provided gradient."""

        NO_DERIVATIVE = "no_derivative"
        """No derivative computation."""

        COMPLEX_STEP = ApproximationMode.COMPLEX_STEP
        """Complex-step approximation."""

        FINITE_DIFFERENCES = ApproximationMode.FINITE_DIFFERENCES
        """Finite differences approximation."""

        CENTERED_DIFFERENCES = ApproximationMode.CENTERED_DIFFERENCES
        """Centered differences approximation."""

    def __init__(
        self,
        input_space: _SpaceT,
        database: Database | None = None,
        differentiation_method: DifferentiationMethod = DifferentiationMethod.USER,
        differentiation_step: float = 1e-7,
        parallel_differentiation: bool = False,
    ) -> None:
        """
        Args:
            input_space: The input space on which the functions are evaluated.
            database: The initial database to store the function evaluations.
                If `None`,
                the problem starts from an empty database.
                If there is no need to store the function evaluations,
                this argument is ignored.
            differentiation_method: The differentiation method
                to evaluate the derivatives.
            differentiation_step: The step used by the differentiation method,
                taken in the coordinates the algorithm works in.
                This argument is ignored
                when the differentiation method is not an
                [ApproximationMode][gemseo.core.problem.evaluation.EvaluationProblem.ApproximationMode].
            parallel_differentiation: Whether
                to approximate the derivatives in parallel.
        """  # noqa: D205, D212, D415
        self.__observables = Observables()
        self.__new_iter_observables = Observables()
        self.differentiation_step = differentiation_step
        self.differentiation_method = differentiation_method
        self._database = (
            Database(input_space=input_space) if database is None else database
        )
        self._input_space = input_space
        self.__initial_current_value = deepcopy(input_space._current_value)
        self._stop_if_nan = True
        self.__parallel_differentiation = parallel_differentiation
        self.__parallel_differentiation_options = {}
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
        mls.add(
            "Evaluate the functions: {}",
            pretty_str(self.function_names, use_and=False),
        )
        return mls

    @property
    def input_space(self) -> _SpaceT:
        """The input space on which the functions are evaluated."""
        return self._input_space

    @property
    def database(self) -> Database:
        """The database to store the function evaluations."""
        return self._database

    @database.setter
    def database(self, database: Database):
        self._database = database

    def __iter_functions(self) -> Iterator[ArrayFunction]:
        """Iterate over every function of the problem.

        Unlike
        [functions][gemseo.core.problem.evaluation.EvaluationProblem.functions],
        this walks the new-iteration observables too,
        and the functions a subclass holds under a name of its own,
        skipping a name that holds none.

        Yields:
            The functions.
        """
        for functions in self._sequence_of_functions:
            yield from functions

        for function_name in self._function_names:
            function = getattr(self, function_name)
            if function is not None:
                yield function

    def _release_transformation(self) -> None:
        """Release the perturbation map of every evaluation half of this problem.

        Undoes what
        [create_working_problem][gemseo.core.problem.evaluation.EvaluationProblem.create_working_problem]
        told the
        [EvaluationFunction][gemseo.core.function.evaluation_function.EvaluationFunction]
        of every function this problem holds,
        objective, constraints, observables and new-iteration observables alike:
        once this returns,
        an approximated Jacobian of any of them perturbs
        in the coordinates the user declared again,
        instead of the working ones of the last run.
        A [BaseDriverLibrary][gemseo.core.algorithm.base_driver_library.BaseDriverLibrary]
        calls this in its `finally` block,
        so it runs whether the run succeeds or raises.
        """  # noqa: E501
        for function in self.__iter_functions():
            if isinstance(function, EvaluationFunction):
                function.set_perturbation_transformation(None)

    @property
    def stop_if_nan(self) -> bool:
        """Whether the evaluation stops when a function returns `NaN`."""
        return self._stop_if_nan

    @stop_if_nan.setter
    def stop_if_nan(self, value: bool) -> None:
        self._stop_if_nan = value
        for function in self.__iter_functions():
            if isinstance(function, BaseProblemFunction):
                function.stop_if_nan = value

    @property
    def parallel_differentiation(self) -> bool:
        """Whether to approximate the derivatives in parallel.

        This attribute is ignored
        when the differentiation method is not an
        [ApproximationMode][gemseo.core.problem.evaluation.EvaluationProblem.ApproximationMode].
        """
        return self.__parallel_differentiation

    @parallel_differentiation.setter
    def parallel_differentiation(self, value: bool) -> None:
        self.__parallel_differentiation = value

    @property
    def parallel_differentiation_options(self) -> dict[str, int | bool]:
        """The options to approximate the derivatives in parallel.

        This attribute is ignored
        when the differentiation method is not an
        [ApproximationMode][gemseo.core.problem.evaluation.EvaluationProblem.ApproximationMode].
        """
        return self.__parallel_differentiation_options

    @parallel_differentiation_options.setter
    def parallel_differentiation_options(self, value: dict[str, int | bool]) -> None:
        self.__parallel_differentiation_options = value

    @property
    def observables(self) -> Observables:
        """The observables."""
        return self.__observables

    @observables.setter
    def observables(self, functions: Iterable[ArrayFunction]) -> None:
        self.__observables.clear()
        self.__observables.extend(functions)

    @property
    def new_iter_observables(self) -> Observables:
        """The observables to be evaluated whenever a database entry is created."""
        return self.__new_iter_observables

    @new_iter_observables.setter
    def new_iter_observables(self, functions: Iterable[ArrayFunction]) -> None:
        self.__new_iter_observables.clear()
        self.__new_iter_observables.extend(functions)

    def add_observable(
        self,
        observable: ArrayFunction,
        new_iter: bool = True,
    ) -> None:
        """Add an observable function.

        It is an [ArrayFunction][gemseo.core.function.array_function.ArrayFunction]
        with
        [ArrayFunction.FunctionType.OBS][gemseo.core.function.array_function.ArrayFunction.FunctionType]
        as function type.

        Args:
            observable: The observable function.
            new_iter: Whether to call the observable
                whenever a database entry is created.
        """
        formatted_observable = self.__observables.format(observable)
        if formatted_observable is None:
            return

        self._check_function_name(formatted_observable)
        self.__observables.append(formatted_observable)
        if new_iter:
            self.__new_iter_observables.append(formatted_observable)

    @property
    def functions(self) -> list[ArrayFunction]:
        """All the functions except the "new iter" observables."""
        return list(self.__observables)

    @property
    def original_functions(self) -> list[ArrayFunction]:
        """All the original functions except those of the "new iter" observables."""
        return list(self.__observables.get_originals())

    @property
    def function_names(self) -> list[str]:
        """All the function names except those of the "new iter" observables."""
        return [function.name for function in self.functions]

    def _check_function_name(self, function: ArrayFunction) -> None:
        """Check that the function has a valid name.

        Args:
            function: The function to check.

        Raises:
            ValueError: If the function name is already used.
        """
        if function.name in [
            function_.name for function_ in self.functions if function_ is not None
        ]:
            msg = (
                f"The function name '{function.name}' is already used by another"
                f" function. Duplicated function names produce unpredictable behavior."
            )
            raise ValueError(msg)

    def add_listener(
        self,
        listener: Callable[[RealArray], Any],
        at_each_iteration: bool = True,
        at_each_function_call: bool = False,
        output_names: Iterable[str] = (),
    ) -> None:
        """Add a listener for some events.

        A listener is a function registered on the database
        and called every time one of the events it is registered for occurs,
        until the listeners are cleared.

        Args:
            listener: A function to be called after some events,
                whose argument is an input value.
            at_each_iteration: Whether to evaluate the listeners
                after evaluating all functions
                for a given point and storing their values in the
                [database][gemseo.core.problem.evaluation.EvaluationProblem.database].
            at_each_function_call: Whether to evaluate the listeners
                after storing any new value in the
                [database][gemseo.core.problem.evaluation.EvaluationProblem.database].
            output_names: The names of the output variables
                whose values are to be stored in the database by this listener.
        """
        if at_each_function_call:
            self.database.add_store_listener(listener, output_names=output_names)
        if at_each_iteration:
            self.database.add_new_iter_listener(listener, output_names=output_names)

    def get_functions(
        self,
        no_db_no_norm: bool = False,
        observable_names: Iterable[str] | None = None,
        jacobian_names: Iterable[str] | None = None,
    ) -> tuple[list[ArrayFunction], list[ArrayFunction]]:
        """Return the functions to be evaluated.

        Args:
            no_db_no_norm: Whether to prevent
                both database backup and input value normalization.
            observable_names: The names of the observables to evaluate.
                If empty,
                then all the observables are evaluated.
                If `None`,
                then no observable is evaluated.
            jacobian_names: The names of the functions
                whose Jacobian matrices must be computed.
                If empty,
                then compute the Jacobian matrices of the functions
                that are selected for evaluation using the other arguments.
                If `None`,
                then no Jacobian matrices is computed.

        Returns:
            The functions computing the outputs
            and the functions computing the Jacobians.

        Raises:
            ValueError: If a name in `jacobian_names` is not the name of
                a function of the problem.
        """
        output_functions = self._get_output_functions(no_db_no_norm, observable_names)
        return self._get_output_and_jacobian_functions(
            jacobian_names, output_functions, no_db_no_norm
        )

    def _get_output_and_jacobian_functions(
        self,
        jacobian_names: Iterable[str],
        output_functions: list[ArrayFunction],
        no_db_no_norm: bool,
    ) -> tuple[list[ArrayFunction], list[ArrayFunction]]:
        """Return the output and Jacobian functions to be evaluated.

        Args:
            jacobian_names: The names of the Jacobian functions.
            output_functions: The names of the output functions.
            no_db_no_norm: Whether to prevent
                both database backup and input value normalization.

        Returns:
            The output and Jacobian functions to be evaluated.
        """
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
        jacobian_functions = self._get_output_functions(
            no_db_no_norm,
            observable_names or None,
            **self._get_options_for_get_functions(jacobian_names),
        )
        return output_functions, jacobian_functions

    def evaluate_functions(
        self,
        input_value: RealArray | None = None,
        input_value_is_normalized: bool = True,
        preprocess_input_value: bool = True,
        output_functions: Iterable[ArrayFunction] | None = (),
        jacobian_functions: Iterable[ArrayFunction] | None = None,
    ) -> EvaluationType:
        """Evaluate the functions, and possibly their derivatives.

        Args:
            input_value: The input value at which to evaluate the functions;
                if `None`, use the current value of the input space.
            input_value_is_normalized: Whether `input_value` is normalized.
            preprocess_input_value: Whether to preprocess the input value.
            output_functions: The functions computing the outputs.
                If empty, evaluate all the functions computing outputs.
                If `None`, do not evaluate functions computing outputs.
            jacobian_functions: The functions computing the Jacobians.
                If empty, evaluate all the functions computing Jacobians.
                If `None`, do not evaluate functions computing Jacobians.

        Returns:
            The output values of the functions,
            as well as their Jacobian matrices if `jacobian_functions` is empty.

        Raises:
            ValueError: When `preprocess_input_value` is `True`
                and the input space cannot be normalized
                while `input_value` is `None` or `input_value_is_normalized` is `True`.
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

        if preprocess_input_value:
            functions = output_functions or jacobian_functions
            if functions:
                # N.B. either all functions expect normalized inputs or none of them do.
                input_value = self._preprocess_inputs(
                    input_value,
                    input_value_is_normalized,
                    functions[0].expects_normalized_inputs,
                )

        outputs = {}
        for function in output_functions:
            try:
                outputs[function.name] = function.evaluate(input_value)
            except ValueError:  # noqa: PERF203
                logger.exception("Failed to evaluate function %s", function.name)
                raise

        if not jacobian_functions:
            return outputs, {}

        jacobians = {}
        for function in jacobian_functions:
            try:
                jacobians[function.name] = function.jac(input_value)
            except ValueError:  # noqa: PERF203
                logger.exception("Failed to evaluate Jacobian of %s.", function.name)
                raise

        return outputs, jacobians

    def _get_options_for_get_functions(
        self, jacobian_names: list[str]
    ) -> dict[str, Any]:
        """Return the options for `_get_functions()`.

        Args:
            jacobian_names: The names of the functions
                whose Jacobian matrices must be computed.

        Returns:
            The options for `_get_functions()`.
        """
        return {}

    def _preprocess_inputs(
        self,
        input_value: RealArray | None,
        normalized: bool,
        normalization_expected: bool,
    ) -> RealArray:
        """Prepare the input value for the function evaluation.

        Args:
            input_value: The input value.
                If `None`, use the current value of the input space.
            normalized: Whether the input value is normalized.
            normalization_expected: Whether the functions expect normalized variables.

        Returns:
            The prepared input value.

        Raises:
            ValueError: When the input space does not support normalization
                and the input value is either `None` or normalized.
        """
        input_space = self._input_space
        if not input_space._supports_normalization:
            # Membership and normalization are specific to a design space.
            # The bounds of a random variable are the limits of the support
            # of its probability distribution; they are descriptive only,
            # so the membership of the input value is not checked.
            # A driver passes the input value explicitly, non-normalized,
            # so there is nothing to prepare.
            if input_value is None:
                msg = (
                    "The input value cannot be None "
                    f"because a {input_space.__class__.__name__} "
                    "has no current value."
                )
                raise ValueError(msg)

            if normalized:
                msg = (
                    "The input value cannot be normalized "
                    f"because a {input_space.__class__.__name__} "
                    "cannot be normalized."
                )
                raise ValueError(msg)

            return input_value

        design_space = cast("DesignSpace", input_space)
        if input_value is None:
            input_value = design_space.get_current_value(normalize=normalized)
        elif self.check_bounds:
            if normalized:
                non_normalized_variables = design_space.denormalize_vect(
                    input_value, no_check=True
                )
            else:
                non_normalized_variables = input_value

            design_space.check_membership(non_normalized_variables)

        if normalized and not normalization_expected:
            return design_space.denormalize_vect(input_value, no_check=True)

        if not normalized and normalization_expected:
            return design_space.normalize_vect(input_value)

        return input_value

    def _get_output_functions(
        self,
        no_db_no_norm: bool,
        observable_names: Iterable[str] | None,
    ) -> list[ArrayFunction]:
        """Return functions.

        Args:
            no_db_no_norm: Whether to prevent
                both database backup and input value normalization.
            observable_names: The names of the observables to return.
                If empty,
                then all the observables are returned.
                If `None`,
                then no observable is returned.

        Returns:
            The functions.
        """
        from_original_functions = no_db_no_norm
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

    def bind_functions(
        self,
        use_database: bool = True,
        store_jacobian: bool = True,
        support_sparse_jacobian: bool = False,
        evaluate_observable_jacobian: bool = False,
        vectorize: bool = False,
    ) -> None:
        """Bind each function of the problem to its database and its counters.

        This half evaluates the functions of the problem,
        binding them to its database and its counters,
        and it is the only wrapping the problem performs on itself.
        It works in the coordinates the user declared;
        adapting a point of the working coordinates is the business of
        [create_working_problem][gemseo.core.problem.evaluation.EvaluationProblem.create_working_problem].

        Calling this method again rebuilds the wrappers from the original functions,
        so a second driver run honours its own settings instead of the first's.

        Args:
            use_database: Whether to look up and store the evaluations.
                This governs the database of this problem and nothing else.
            store_jacobian: Whether to record the Jacobian matrices.
            support_sparse_jacobian: Whether the driver supports a sparse Jacobian.
                When it does not, a sparse Jacobian is densified before being
                recorded,
                so the database holds what the driver reads.
            evaluate_observable_jacobian: Whether to evaluate the Jacobian of the
                observables evaluated at each new iteration.
            vectorize: Whether the functions are evaluated on a matrix of samples
                rather than on a single point,
                which a vectorized DOE asks for.

        Raises:
            ValueError: When a function expects normalized inputs,
                which this half evaluates
                in the coordinates the user declared.
        """
        for function in self.functions:
            if function.expects_normalized_inputs:
                # This half works in the coordinates the user declared,
                # whatever the space,
                # and the wrapper it installs takes its input there;
                # normalizing a point is the business of the other half,
                # which wraps this one and cannot hand it a normalized point back.
                # Evaluating such a function at a point of the space instead
                # would silently return a value that is not the one asked for.
                msg = (
                    f"The function {function.name} expects normalized inputs "
                    "while the evaluations are recorded in the coordinates "
                    "the input space declares."
                )
                raise ValueError(msg)

        database = self.database if use_database else None
        # Anything that is not one of the methods computing the derivatives
        # themselves is handed to the factory,
        # which is where an unknown name is refused
        # rather than silently approximating nothing.
        differentiation_method = (
            None
            if self.differentiation_method
            in set(self.DifferentiationMethod).difference(set(self.ApproximationMode))
            else self.differentiation_method
        )
        differentiation_method_options = {
            # Taken in the coordinates the half perturbs in:
            # the user's here,
            # the working ones once `create_working_problem` tells the half the map.
            "step": self.differentiation_step,
            # The half hands the approximator the space it perturbs in,
            # so a component is compared against the bounds of that space
            # as they are.
            "normalize": False,
            "parallel": self.__parallel_differentiation,
            **self.__parallel_differentiation_options,
        }
        input_space = self._input_space
        # The perturbations of an approximated Jacobian are bounded
        # by the input space only when it supports normalization.
        design_space = (
            cast("DesignSpace", input_space)
            if input_space._supports_normalization
            else None
        )

        def bind(function: ArrayFunction, on_samples: bool) -> EvaluationFunction:
            """Bind a function to the database and the counters of the problem.

            Args:
                function: The function to bind.
                on_samples: Whether the function is evaluated on a matrix of samples
                    rather than on a single point.

            Returns:
                The evaluation function.
            """
            # Rebuilt past a wrapper this problem installs itself,
            # so that calling this method again honours the new settings
            # rather than stacking a second one.
            # Any other wrapper is kept:
            # a caller that passed one meant it,
            # as a sub-problem built on the functions of the problem it comes from,
            # whose evaluations have to reach that problem's database too.
            # The wrapper is recognized by the problem that installed it
            # rather than by its database,
            # which a caller may replace with a new one between two runs,
            # or by its type, which cannot tell the two apart.
            owned = isinstance(function, EvaluationFunction) and function._owner is self
            wrapped = function._wrapped_function if owned else function
            new_function = EvaluationFunction(
                wrapped,
                database,
                store_jacobian=store_jacobian,
                support_sparse_jacobian=support_sparse_jacobian,
                vectorize=on_samples,
                design_space=design_space,
                stop_if_nan=self.stop_if_nan,
                owner=self,
                differentiation_method=differentiation_method,
                differentiation_method_options=differentiation_method_options,
            )
            if owned and EvaluationFunction.enable_statistics:
                # Rebuilding past our own wrapper starts a new counter,
                # so a second run would otherwise only ever count itself,
                # even though nothing asked for the previous count to be
                # dropped: that is what `reset(function_calls=True)` is for.
                new_function.n_calls = function.n_calls
            return new_function

        for functions in self._sequence_of_functions:
            # Vectorization does not make sense for the observables evaluated at
            # each new iteration, since they are evaluated point by point.
            on_samples = vectorize and functions is not self.__new_iter_observables
            for index, function in enumerate(functions):
                functions[index] = bind(function, on_samples)

        for function_name in self._function_names:
            function = getattr(self, function_name)
            if function is not None:
                setattr(self, function_name, bind(function, vectorize))

        self.new_iter_observables.evaluate_jacobian = evaluate_observable_jacobian

    def __check_transformation(self, transformation: BaseSpaceTransformation) -> None:
        """Check that the transformation is built for the input space of this problem.

        Args:
            transformation: The map from the original coordinates to the working ones.

        Raises:
            ValueError: When the transformation is not built for the input space.
        """
        # An identity comparison:
        # the transformation reads the bounds of the very space it was built for,
        # and an equal copy of the input space is still another space.
        if transformation.original_space is not self._input_space:
            msg = (
                f"The {type(transformation).__name__} is not built for the input "
                "space of the problem; build it for that space."
            )
            raise ValueError(msg)

    def __check_functions_are_bound(self) -> None:
        """Check that the evaluation half is installed on every function.

        The adaptation half is built on the evaluation one:
        it maps a point back to the coordinates the user declared
        and hands it to that half,
        which is what records the evaluation and counts it.
        Adapting a function the problem has not bound yet
        would therefore build a problem
        whose run reaches the user's function and leaves no history behind.

        Raises:
            ValueError: When a function of the problem has not been bound.
        """
        names = [
            function.name
            for function in self.functions
            if not isinstance(function, BaseProblemFunction)
        ]
        if names:
            msg = (
                "The following functions have not been bound: "
                f"{pretty_str(names)}; "
                "call bind_functions before create_working_problem."
            )
            raise ValueError(msg)

    def create_working_problem(
        self,
        transformation: BaseSpaceTransformation,
    ) -> EvaluationProblem:
        """Derive the working problem, evaluated in the working coordinates.

        The functions of this problem must have been bound by
        [bind_functions][gemseo.core.problem.evaluation.EvaluationProblem.bind_functions]
        beforehand,
        since the half installed here is built on the one installed there.

        The functions, the database and the input space of the problem this method
        is called on keep the ones the user built: the working problem is a
        **new** one, over the working space, and disposable,
        a driver building one for the duration of a run.
        Calling this method does change two things of the problem it is called on,
        both meant to be undone once the run is over:
        [working_database][gemseo.core.problem.evaluation.EvaluationProblem.working_database]
        is set, and kept afterwards, since it is a debugging aid read once the run
        has ended, and the evaluation halves of its functions are told the
        transformation, as described below.
        A [BaseDriverLibrary][gemseo.core.algorithm.base_driver_library.BaseDriverLibrary]
        releases that map in its `finally` block,
        through the private `_release_transformation()`,
        so it runs whether the run succeeds or raises
        and the user's own functions perturb in their own coordinates again
        once it returns.

        The database is **shared**, not copied,
        since the history, the KKT checker and the progress bar read it
        while the run goes on.
        When
        [enable_working_database][gemseo.core.problem.evaluation.EvaluationProblem.enable_working_database]
        is set,
        a second one is created, in the working coordinates,
        and is reachable from both problems as `working_database`.

        The evaluation halves inside the functions of this problem
        are the very ones this problem holds, not copies,
        and are told the map through
        [set_perturbation_transformation][gemseo.core.function.evaluation_function.EvaluationFunction.set_perturbation_transformation]:
        an approximated Jacobian asked of either problem therefore perturbs
        in the working coordinates
        and is recorded in the coordinates the user declared.
        Calling
        [bind_functions][gemseo.core.problem.evaluation.EvaluationProblem.bind_functions]
        again rebuilds them without it.

        Args:
            transformation: The map from the original coordinates to the working ones.

        Returns:
            The working problem, whose functions wrap the ones of this problem.

        Raises:
            ValueError: When a function of the problem has not been bound
                or when the transformation is not built for the input space.
        """  # noqa: E501
        self.__check_functions_are_bound()
        self.__check_transformation(transformation)
        working_space = transformation.working_space
        # Set before the copy below,
        # so that the problem the user keeps
        # and the disposable one a driver runs read the same store.
        # The assignment is total:
        # a run made after the setting was turned off
        # must not keep writing into the store of an earlier one.
        self.working_database = (
            Database(name=f"{self.database.name}_working", input_space=working_space)
            if self.enable_working_database
            else None
        )

        # The working problem is this one
        # with its input space and its functions swapped,
        # which a copy expresses and a constructor cannot:
        # a subclass is free to define its own signature,
        # as every benchmark problem does.
        working_problem = copy(self)
        working_problem._input_space = working_space

        def adapt(function: BaseProblemFunction) -> TransformedInputFunction:
            """Adapt the argument of a function to the working coordinates.

            Args:
                function: The half to adapt.

            Returns:
                The adapted function.
            """
            if isinstance(function, EvaluationFunction):
                # Told rather than copied:
                # the hooks, the counters and stop_if_nan stay shared,
                # and a stacked adaptation half keeps the map it was told.
                function.set_perturbation_transformation(transformation)

            return TransformedInputFunction(
                function,
                transformation,
                stop_if_nan=self.stop_if_nan,
                working_database=self.working_database,
            )

        # A collection is copied
        # rather than built, for the same reason,
        # and its functions are replaced directly:
        # going through the `add_*` methods of the problem
        # would format them a second time,
        # offsetting an already offset constraint.
        working_collections = []
        collection_id_to_working = {}
        for collection in self._sequence_of_functions:
            working_collection = copy(collection)
            if collection is self.__new_iter_observables:
                # These are evaluated from inside a database store notification,
                # at the point just recorded,
                # in the coordinates the user declared.
                # They need the recording they already carry and no adaptation:
                # adapting them would hand a point of the original space
                # to a map expecting one of the working space.
                # Their perturbations are taken in the working coordinates all
                # the same, since they are told the map too.
                working_collection._functions = list(collection)
                for function in working_collection:
                    if isinstance(function, EvaluationFunction):
                        function.set_perturbation_transformation(transformation)
            else:
                working_collection._functions = [
                    adapt(function) for function in collection
                ]

            working_collections.append(working_collection)
            collection_id_to_working[id(collection)] = working_collection

        working_problem._sequence_of_functions = working_collections
        # A collection is reachable both from `_sequence_of_functions`
        # and from an attribute of the problem,
        # so every reference to it is rebound.
        # The attributes are walked once,
        # against every collection at once,
        # and a collection is recognized by identity
        # because a subclass is free to contribute collections of its own,
        # under names unknown here.
        for name, value in self.__dict__.items():
            working_collection = collection_id_to_working.get(id(value))
            if working_collection is not None:
                working_problem.__dict__[name] = working_collection

        for function_name in self._function_names:
            function = getattr(self, function_name)
            if function is not None:
                setattr(working_problem, function_name, adapt(function))

        return working_problem

    def check(self) -> None:
        """Check if the functions attached to the problem can be evaluated."""
        self.input_space.check()

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
    ) -> Dataset:
        """Export the database of the problem to dataset.

        Args:
            name: The name to be given to the dataset.
                If empty,
                use the name of the
                [database][gemseo.core.problem.evaluation.EvaluationProblem.database].
            categorize: Whether to distinguish
                between the different groups of variables.
                If so,
                use an [IODataset][gemseo.dataset.io_dataset.IODataset]
                with the input variables in the
                [input_group][gemseo.dataset.io_dataset.IODataset.input_group]
                and the functions and their derivatives
                in the
                [output_group][gemseo.dataset.io_dataset.IODataset.output_group].
                Otherwise,
                group all the variables in
                [parameter_group][gemseo.dataset.dataset.Dataset.parameter_group].
            export_gradients: Whether to export the gradients of the functions
                if the latter are available in the database of the problem.
            input_values: The input values to be considered.
                If empty, consider all the input values of the database.

        Returns:
            A dataset built from the database of the problem.
        """
        if categorize:
            dataset_class = IODataset
            input_group = IODataset.input_group
            output_group = IODataset.output_group
            gradient_group = Dataset.gradient_group
        else:
            dataset_class = Dataset
            input_group = output_group = gradient_group = Dataset.default_group

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
        input_space: bool = True,
        function_calls: bool = True,
    ) -> None:
        """Partially or fully reset the problem.

        Args:
            database: Whether to clear the database.
            current_iter: Whether to reset the counter of evaluations
                to the initial iteration.
            input_space: Whether to restore the current value that the input space
                had at the instantiation of the problem.
            function_calls: Whether to reset the number of calls of the functions.
        """
        if current_iter:
            self.evaluation_counter.current = 0
            # This is the start of an evaluation process.
            # Disable the evaluation counter
            # to prevent the driver from finalizing the previous iteration.
            self.evaluation_counter.enabled = False

        if database:
            self.database.clear()

        if input_space:
            self.input_space._current_value = self.__initial_current_value

        if function_calls and EvaluationFunction.enable_statistics:
            # A function is its own original until the functions are bound,
            # so resetting its original too is a no-op on a problem no driver has run.
            for function in self.__iter_functions():
                function.n_calls = function.original.n_calls = 0

    def to_hdf(
        self,
        file_path: StrPath,
        append: bool = False,
        hdf_node_path: str = "",
    ) -> None:
        """Export the evaluation history and results to an HDF file.

        Args:
            file_path: The HDF file path.
            append: Whether to append the data to the file if not empty.
                Otherwise,
                overwrite data.
            hdf_node_path: The path of the HDF node
                in which the evaluation problem should be exported.
                If empty, the root node is considered.
        """
        msg = "Exporting the evaluation problem to the file %s"
        if hdf_node_path:
            logger.info(msg + " at node %s", file_path, hdf_node_path)  # noqa: G003
        else:
            logger.info(msg, file_path)

        self.database.to_hdf(file_path, append=append, hdf_node_path=hdf_node_path)
