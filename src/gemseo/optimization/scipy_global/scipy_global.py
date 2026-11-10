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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The library of SciPy global optimization algorithms."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import replace
from functools import partial
from math import ceil
from os import getpid
from pathlib import Path
from pickle import dumps
from sys import maxsize
from traceback import format_exc
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from numpy import clip
from numpy import float64
from numpy import inf as np_inf
from numpy import int32
from numpy import isfinite
from numpy import mod
from numpy import real
from numpy.random import default_rng
from numpy.typing import NDArray
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
from scipy.optimize import basinhopping
from scipy.optimize import differential_evolution
from scipy.optimize import direct
from scipy.optimize import dual_annealing
from scipy.optimize import shgo

from gemseo.optimization.core.base_optimization_library import BaseOptimizationLibrary
from gemseo.optimization.core.base_optimization_library import (
    OptimizationAlgorithmDescription,
)
from gemseo.optimization.scipy_global.settings.base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.optimization.scipy_global.settings.basinhopping import BASINHOPPING_Settings
from gemseo.optimization.scipy_global.settings.basinhopping_local_method import (
    BasinHoppingLocalMethod,
)
from gemseo.optimization.scipy_global.settings.differential_evolution import (
    DIFFERENTIAL_EVOLUTION_Settings,
)
from gemseo.optimization.scipy_global.settings.differential_evolution import (
    DifferentialEvolutionUpdating,
)
from gemseo.optimization.scipy_global.settings.direct import DIRECT_Settings
from gemseo.optimization.scipy_global.settings.dual_annealing import (
    DUAL_ANNEALING_Settings,
)
from gemseo.optimization.scipy_global.settings.shgo import SHGO_Settings
from gemseo.space.util import get_value_and_bounds
from gemseo.util._worker_context import tag_process_worker
from gemseo.util.constant import n_cpus
from gemseo.util.multiprocessing import start_method
from gemseo.util.multiprocessing.process_pool import create_process_pool_executor
from gemseo.util.multiprocessing.process_pool import get_number_of_processes

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from concurrent.futures import ProcessPoolExecutor

    from numpy.random import Generator

    from gemseo.core.function.array_function import ArrayFunction
    from gemseo.core.function.array_function import OutputType
    from gemseo.core.function.array_function import WrappedFunctionType
    from gemseo.core.function.array_function import WrappedJacobianType
    from gemseo.core.problem.database import DatabaseValueType
    from gemseo.optimization.problem import OptimizationProblem

InputType = NDArray[float64 | int32]

logger = logging.getLogger(__name__)

_worker_library: ScipyGlobalOpt | None = None
"""The library stashed in a worker process by the initializer of the pool.

This module global is per worker process:
it is set once, at the start of the process,
so that a task does not carry the library.
"""


def _init_worker(library: ScipyGlobalOpt, parent_id: int, parent_path: Path) -> None:
    """Initialize a worker process of the pool of `ScipyGlobalOpt`.

    Stash the copy of the library of this process,
    tag the process with its parent metadata
    as the processes of a parallel execution are,
    and stop listening to the databases of the copy
    as the calling process counts the iterations when storing the entries.

    Args:
        library: The library to stash in this process.
        parent_id: The process id of the parent process.
        parent_path: The working directory of the parent at submission time.
    """
    global _worker_library
    _worker_library = library
    tag_process_worker(parent_id, parent_path)
    library._progress_bar = None
    problem = library._original_problem
    for function in problem.functions:
        function.pre_compute_at_new_point = None
        # The calling process counts the calls when it reads the values back,
        # and with the fork start method,
        # the counter of a copy is the shared memory of the original function.
        function.enable_statistics = False

    problem.database.clear_listeners()
    working_database = library._problem.working_database
    if working_database is not None:
        working_database.clear_listeners()


def _evaluate_in_worker(
    task: tuple[InputType, NDArray[float64], DatabaseValueType],
) -> tuple[list[tuple[NDArray[float64], DatabaseValueType]], Exception | None]:
    """Evaluate the objective and the constraints at a point in a worker process.

    An exception raised by an evaluation is returned, not raised,
    together with the entries stored before it:
    `ProcessPoolExecutor.map` would otherwise drop the results
    of the other points of the chunk.
    An exception that cannot be pickled would hide the error
    behind the one raised by the pool when sending it back,
    so it is replaced with a `RuntimeError`
    carrying its type name, its message and its traceback.

    Args:
        task: The point at which to evaluate the functions,
            in the working coordinates,
            the same point in the original coordinates,
            and the outputs already stored in the database for this point,
            which are not computed again.

    Returns:
        The entries the evaluations added to the database of the problem,
        which is emptied beforehand to hold only this task,
        and the exception raised by an evaluation, if any.
    """
    input_value, original_input_value, known_outputs = task
    library = _worker_library
    database = library._original_problem.database
    database.clear()
    if known_outputs:
        database.store(original_input_value, known_outputs)

    working_database = library._problem.working_database
    if working_database is not None:
        working_database.clear()

    problem = library._problem
    error = None
    try:
        problem.objective.evaluate(input_value)
        for constraint in problem.constraints:
            constraint.evaluate(input_value)
    except Exception as exception:  # noqa: BLE001
        traceback = format_exc()
        exception.add_note(f"Raised in a worker process:\n{traceback}")
        try:
            dumps(exception)
        except Exception:  # noqa: BLE001
            exception = RuntimeError(
                f"{type(exception).__name__}: {exception}\n{traceback}"
            )

        error = exception

    entries = [(key.wrapped_array, dict(outputs)) for key, outputs in database.items()]
    return entries, error


class _BoundedRandomDisplacement:
    """A random displacement for basin-hopping that stays inside the bounds.

    A displaced point that leaves the bounds is reflected on them,
    as a ball bouncing on the walls of a box,
    instead of being projected onto them,
    which would pile up the points on the faces of the box.
    """

    stepsize: float
    """The maximum displacement along each coordinate, adapted by SciPy."""

    def __init__(
        self,
        stepsize: float,
        rng: Generator,
        lower_bound: NDArray[float64],
        upper_bound: NDArray[float64],
    ) -> None:
        """
        Args:
            stepsize: The initial maximum displacement along each coordinate.
            rng: The random number generator.
            lower_bound: The lower bounds, `-inf` when a component is unbounded.
            upper_bound: The upper bounds, `inf` when a component is unbounded.
        """  # noqa: D205, D212, D415
        self.stepsize = stepsize
        self._rng = rng
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        # The components for which the reflection is a folding into a box.
        self._box = (
            isfinite(lower_bound) & isfinite(upper_bound) & (upper_bound > lower_bound)
        )

    def __call__(self, x_vect: InputType) -> InputType:
        """Displace a point randomly and reflect it on the bounds.

        Args:
            x_vect: The point to displace.

        Returns:
            The displaced point.
        """
        lower_bound = self._lower_bound
        upper_bound = self._upper_bound
        y_vect = x_vect + self._rng.uniform(-self.stepsize, self.stepsize, x_vect.shape)
        box = self._box
        if box.any():
            # Fold into the box with a triangle wave, whatever the step length.
            lower = lower_bound[box]
            width = upper_bound[box] - lower
            y_vect[box] = (
                lower + width - abs(mod(y_vect[box] - lower, 2 * width) - width)
            )

        # Reflect once on a finite bound that is crossed, for the other components.
        other = ~box
        below = other & (y_vect < lower_bound)
        y_vect[below] = 2 * lower_bound[below] - y_vect[below]
        above = other & (y_vect > upper_bound)
        y_vect[above] = 2 * upper_bound[above] - y_vect[above]
        # A safety net for the fixed variables and the one-sided bounds.
        return clip(y_vect, lower_bound, upper_bound)


@dataclass
class SciPyGlobalAlgorithmDescription(OptimizationAlgorithmDescription):
    """The description of a global optimization algorithm from the SciPy library."""

    library_name: str = "SciPy Global Optimization"
    """The library name."""

    settings_class: type[BaseSciPyGlobalSettings] = BaseSciPyGlobalSettings
    """The option validation model for SciPy global optimization library."""


class ScipyGlobalOpt(BaseOptimizationLibrary[BaseSciPyGlobalSettings]):
    """The library of SciPy global optimization algorithms."""

    __names_to_functions: ClassVar[Mapping[str, Callable]] = MappingProxyType({
        "BASINHOPPING": basinhopping,
        "DIFFERENTIAL_EVOLUTION": differential_evolution,
        "DIRECT": direct,
        "DUAL_ANNEALING": dual_annealing,
        "SHGO": shgo,
    })
    """The mapping between the algorithm name and the SciPy function."""

    __doc: Final[str] = "https://docs.scipy.org/doc/scipy/reference/generated/"

    __max_direct_evaluations: Final[int] = 10_000_000
    """The cap on the evaluations passed to DIRECT.

    SciPy's DIRECT allocates memory proportionally to this budget
    and fails when it is too large,
    while GEMSEO monitors the budget through `max_iter`.
    """

    __local_methods_with_gradient: Final[frozenset[BasinHoppingLocalMethod]] = (
        frozenset({
            BasinHoppingLocalMethod.L_BFGS_B,
            BasinHoppingLocalMethod.SLSQP,
            BasinHoppingLocalMethod.TNC,
            BasinHoppingLocalMethod.TRUST_CONSTR,
        })
    )
    """The local methods of basin-hopping using the gradient of the objective."""

    __local_methods_with_equality_constraints: Final[
        frozenset[BasinHoppingLocalMethod]
    ] = frozenset({
        BasinHoppingLocalMethod.COBYQA,
        BasinHoppingLocalMethod.SLSQP,
        BasinHoppingLocalMethod.TRUST_CONSTR,
    })
    """The local methods of basin-hopping handling equality constraints."""

    __local_methods_with_inequality_constraints: Final[
        frozenset[BasinHoppingLocalMethod]
    ] = __local_methods_with_equality_constraints | {BasinHoppingLocalMethod.COBYLA}
    """The local methods of basin-hopping handling inequality constraints."""

    __basinhopping_local_method: BasinHoppingLocalMethod | None = None
    """The local method of basin-hopping for the current run, if any.

    It is chosen once per run, by `_check_algorithm`,
    from the problem of the user,
    as `_pre_run` can then replace its functions with functions
    reporting a Jacobian whatever the original ones,
    e.g. the scaled ones of `scaling_threshold`.
    Before `_check_algorithm` and after `_reset`, it is `None`
    and `_get_algorithm_description_to_check` returns the fixed description.
    """

    ALGORITHM_INFOS: ClassVar[dict[str, SciPyGlobalAlgorithmDescription]] = {
        "BASINHOPPING": SciPyGlobalAlgorithmDescription(
            algorithm_name="Basin-hopping",
            description="Basin-hopping",
            # The default local method handles both kinds of constraints;
            # a local method set by the user may handle fewer,
            # which _get_algorithm_description_to_check accounts for.
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            internal_algorithm_name="basinhopping",
            positive_constraints=True,
            website=f"{__doc}scipy.optimize.basinhopping.html",
            settings_class=BASINHOPPING_Settings,
        ),
        "DIRECT": SciPyGlobalAlgorithmDescription(
            algorithm_name="DIRECT",
            description="DIviding RECTangles",
            handle_unbounded_variables=False,
            internal_algorithm_name="direct",
            website=f"{__doc}scipy.optimize.direct.html",
            settings_class=DIRECT_Settings,
        ),
        "DUAL_ANNEALING": SciPyGlobalAlgorithmDescription(
            algorithm_name="Dual annealing",
            description="Dual annealing",
            internal_algorithm_name="dual_annealing",
            website=f"{__doc}scipy.optimize.dual_annealing.html",
            settings_class=DUAL_ANNEALING_Settings,
        ),
        "SHGO": SciPyGlobalAlgorithmDescription(
            algorithm_name="SHGO",
            description="Simplicial homology global optimization",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            internal_algorithm_name="shgo",
            positive_constraints=True,
            website=f"{__doc}scipy.optimize.shgo.html",
            settings_class=SHGO_Settings,
        ),
        "DIFFERENTIAL_EVOLUTION": SciPyGlobalAlgorithmDescription(
            algorithm_name="Differential evolution",
            description="Differential Evolution algorithm",
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            handle_integer_variables=True,
            internal_algorithm_name="differential_evolution",
            website=f"{__doc}scipy.optimize.differential_evolution.html",
            settings_class=DIFFERENTIAL_EVOLUTION_Settings,
        ),
    }

    def _reset(self) -> None:
        super()._reset()
        self.__basinhopping_local_method = None

    def _pre_run(self, problem: OptimizationProblem) -> None:
        if self._algo_name == "DIRECT":
            max_iter = self._settings.max_iter
            max_evaluations = self.__max_direct_evaluations
            if max_iter > max_evaluations:
                logger.warning(
                    "DIRECT cannot evaluate the functions more than %s times; "
                    "max_iter is reduced from %s to this value.",
                    max_evaluations,
                    max_iter,
                )
                # The reduction precedes the parent method,
                # which builds the iteration observer from `max_iter`;
                # the settings of the user are copied, not modified.
                self._settings = self._settings.model_copy(
                    update={"max_iter": max_evaluations}
                )

        super()._pre_run(problem)

    def _check_algorithm(self, problem: OptimizationProblem) -> None:
        if self._algo_name == "BASINHOPPING":
            self.__basinhopping_local_method = self.__get_local_method(problem)

        super()._check_algorithm(problem)

    def __get_local_method(
        self, problem: OptimizationProblem
    ) -> BasinHoppingLocalMethod:
        """Return the local method of basin-hopping.

        Args:
            problem: The problem to be solved.

        Returns:
            The local method of the settings if any,
            otherwise, when the gradients of the objective and the constraints
            are available or approximated by the `differentiation_method`
            (finite differences or complex step),
            SLSQP for a constrained problem and L-BFGS-B for an unconstrained one,
            and, when they are not,
            COBYLA for a problem with inequality constraints only,
            COBYQA for a problem with equality constraints
            and Nelder-Mead for an unconstrained one.
        """
        local_method = self._settings.local_method
        if local_method is not None:
            return local_method

        constraints = problem.constraints
        # A function wrapped for a previous run, or bound to a problem
        # that is itself a sub-problem, reports a Jacobian
        # whether or not the function of the user has one,
        # hence the use of the fully original functions.
        has_gradient = problem.differentiation_method in set(
            problem.ApproximationMode
        ) or all(
            self.__get_fully_original(function).has_jac
            for function in (problem.objective, *constraints)
        )
        if constraints:
            if has_gradient:
                return BasinHoppingLocalMethod.SLSQP

            # COBYQA is kept for the equality constraints only,
            # as, with inequality constraints given as dictionaries,
            # SciPy's COBYQA can stall or stop away from the optimum.
            if tuple(constraints.get_equality_constraints()):
                return BasinHoppingLocalMethod.COBYQA

            return BasinHoppingLocalMethod.COBYLA

        if has_gradient:
            return BasinHoppingLocalMethod.L_BFGS_B

        return BasinHoppingLocalMethod.NELDER_MEAD

    @staticmethod
    def __get_fully_original(function: ArrayFunction) -> ArrayFunction:
        """Return the function at the end of the chain of wrapped functions.

        Args:
            function: The function.

        Returns:
            The function that does not wrap any other one.
        """
        while function.original is not function:
            function = function.original

        return function

    def _get_algorithm_description_to_check(self) -> SciPyGlobalAlgorithmDescription:
        description = super()._get_algorithm_description_to_check()
        local_method = self.__basinhopping_local_method
        if local_method is None:
            return description

        # The local method of basin-hopping sets what the algorithm handles.
        return replace(
            description,
            handle_equality_constraints=(
                local_method in self.__local_methods_with_equality_constraints
            ),
            handle_inequality_constraints=(
                local_method in self.__local_methods_with_inequality_constraints
            ),
            require_gradient=local_method in self.__local_methods_with_gradient,
        )

    def _evaluate_objective_and_constraints(self, x_vect: InputType) -> None:
        """Evaluate the objective and constraint functions.

        This is a new-iteration listener, registered through
        `problem.add_listener` with its default `at_each_iteration=True`,
        so the point it receives is a key of the database
        and is expressed in the coordinates the user declared.
        The functions of the original problem are the ones that take such a point.

        Args:
            x_vect: The input data with which to call the functions.
        """
        self._original_problem.objective.evaluate(x_vect)
        for constraint in self._original_problem.constraints:
            constraint.evaluate(x_vect)

    def _compute_objective(self, x_vect: InputType) -> OutputType:
        """Wrap the objective function of the problem to pass it to SciPy.

        Cast the result to real.

        Args:
            x_vect: The input data with which to call the function.
        """
        return real(self._problem.objective.evaluate(x_vect))

    def __map(
        self,
        executor: ProcessPoolExecutor,
        n_processes: int,
        func: Callable[[InputType], Any],
        inputs: Iterable[InputType],
    ) -> list[Any]:
        """Evaluate a function at several points in parallel, as SciPy's `workers`.

        The points whose objective or constraints are missing from the database
        are sent to the worker processes,
        each point once and with the outputs already known,
        which evaluate the objective and the constraints on their own copy
        of the library, stashed at the creation of the pool.
        The entries they add to their copy of the database are sent back
        and stored in the database of the problem,
        in the order of the points,
        as if the evaluations had been made in this process;
        an exception raised by an evaluation is raised
        once the entries of the previous points are stored.
        Then the function is evaluated in this process:
        every value is read from the database,
        so nothing is computed again
        and the problem counts one call per point, as when it runs serially.

        Args:
            executor: The pool of worker processes.
            n_processes: The number of worker processes.
            func: The function to evaluate, wrapped by SciPy.
            inputs: The points at which to evaluate the function,
                in the working coordinates.

        Returns:
            The values of the function at these points.
        """
        inputs = list(inputs)
        original_problem = self._original_problem
        database = original_problem.database
        names = [original_problem.objective.name]
        names.extend(constraint.name for constraint in original_problem.constraints)
        inverse_transform_value = self._transformation.inverse_transform_value
        points_to_send = {}
        for point in inputs:
            key = database.get_hashable_ndarray(inverse_transform_value(point))
            if key in points_to_send:
                continue

            outputs = database.get(key)
            if outputs is None or any(name not in outputs for name in names):
                points_to_send[key] = (
                    point,
                    key.wrapped_array,
                    dict(outputs or {}),
                )

        if points_to_send:
            for entries, error in executor.map(
                _evaluate_in_worker,
                points_to_send.values(),
                chunksize=max(1, ceil(len(points_to_send) / (4 * n_processes))),
            ):
                for input_value, outputs in entries:
                    if not database.get(input_value):
                        # This is what the evaluation layer does
                        # before evaluating a function at a new point in this process,
                        # which counts the iterations
                        # and checks the termination criteria;
                        # a point stored with no output is a new point.
                        self._finalize_previous_iteration_using_database()

                    database.store(input_value, outputs)

                if error is not None:
                    raise error

        return [func(point) for point in inputs]

    def _run(self, problem: OptimizationProblem) -> tuple[str, Any]:
        prepare_settings = {
            "BASINHOPPING": self.__prepare_basinhopping,
            "DIFFERENTIAL_EVOLUTION": self.__prepare_differential_evolution,
            "DIRECT": self.__prepare_direct,
            "DUAL_ANNEALING": self.__prepare_dual_annealing,
            "SHGO": self.__prepare_shgo,
        }[self._algo_name]

        # This is required because some algorithms do not
        # call the objective very often when the problem
        # is very constrained (Power2) and OptProblem may fail
        # to detect the optimum.
        # The try block opens right after the listener is registered, so that
        # an exception raised anywhere below, e.g. while building the settings,
        # still reaches the finally clause that removes it.
        executor = None
        completed = False
        try:
            if problem.constraints:
                problem.add_listener(self._evaluate_objective_and_constraints)

            filtered_settings = self._filter_settings()
            # The argument of the SciPy function that follows the objective.
            argument = prepare_settings(problem, filtered_settings)
            workers = filtered_settings.get("workers", 1)
            if workers != 1:
                # SciPy's own pool would lose the databases of the worker processes.
                # The executor is not an attribute
                # as the worker processes receive a copy of this library.
                if self._algo_name == "SHGO":
                    n_tasks = self._settings.n
                else:
                    n_tasks = self._settings.popsize * problem.input_space.dimension

                n_tasks = max(1, n_tasks)
                n_processes = get_number_of_processes(
                    n_cpus if workers == -1 else workers, n_tasks
                )
                # The library is pickled once per worker process,
                # when the process is spawned by a `submit` of the main thread,
                # i.e. while nothing writes into the database.
                executor = create_process_pool_executor(
                    n_processes,
                    start_method.multi_processing_start_method,
                    _init_worker,
                    (self, getpid(), Path.cwd()),
                )
                filtered_settings["workers"] = partial(
                    self.__map, executor, n_processes
                )

            opt_result = self.__names_to_functions[self._algo_name](
                self._compute_objective, argument, **filtered_settings
            )
            completed = True
        finally:
            if executor is not None:
                # When the run stops in the middle of a batch, e.g. on max_iter,
                # the results of the running tasks will not be read,
                # so they are not waited for.
                executor.shutdown(wait=completed, cancel_futures=not completed)

            # Otherwise the listener outlives this run and dereferences
            # `self._original_problem`, which `_reset()` sets back to `None`,
            # so the next store on this problem's database would fail.
            if problem.constraints:
                problem.database.clear_listeners(
                    new_iter_listeners=(self._evaluate_objective_and_constraints,),
                    store_listeners=None,
                )

        message = opt_result.message
        # SciPy's basin-hopping gives its message as a list of strings.
        if isinstance(message, list):
            message = "; ".join(message)

        return message, opt_result.success

    @staticmethod
    def __get_bounds(
        problem: OptimizationProblem,
    ) -> list[tuple[float | None, float | None]]:
        """Return the bounds of the problem in the SciPy format.

        The problem works in the coordinates the algorithm manipulates,
        so its space already carries the bounds the algorithm needs.

        Args:
            problem: The problem to be solved.

        Returns:
            The pairs of lower and upper bounds,
            where an infinite value is replaced with `None`.
        """
        _, lower_bound, upper_bound = get_value_and_bounds(problem.input_space)
        return [
            (
                lower if isfinite(lower) else None,
                upper if isfinite(upper) else None,
            )
            for lower, upper in zip(lower_bound, upper_bound, strict=False)
        ]

    def __prepare_shgo(
        self, problem: OptimizationProblem, settings: MutableMapping[str, Any]
    ) -> list[tuple[float | None, float | None]]:
        """Prepare the settings of SHGO.

        Args:
            problem: The problem to be solved.
            settings: The settings to update in place.

        Returns:
            The bounds.
        """
        settings["constraints"] = self.__get_constraints_as_scipy_dictionary(problem)
        # Deactivate termination criteria which are handled by GEMSEO
        settings["options"].update(
            dict.fromkeys(["maxev", "maxfev", "maxiter", "maxtime"], maxsize)
        )
        settings["options"]["ftol"] = 0.0
        return self.__get_bounds(problem)

    def __prepare_differential_evolution(
        self, problem: OptimizationProblem, settings: MutableMapping[str, Any]
    ) -> list[tuple[float | None, float | None]]:
        """Prepare the settings of differential evolution.

        Args:
            problem: The problem to be solved.
            settings: The settings to update in place.

        Returns:
            The bounds.
        """
        settings["constraints"] = self.__get_non_linear_constraints(problem)
        # The algorithm handles integer variables itself,
        # so the working space still carries their (integral) bounds,
        # as `DesignSpace` normalization leaves integer variables alone;
        # SciPy is told which components are integer so that it keeps
        # them integral at every evaluation instead of exploring them
        # continuously, unlike the discrete ones, which stay unhandled.
        settings["integrality"] = problem.input_space.get_integer_mask()
        workers = settings.get("workers", 1)
        if workers != 1 and problem.constraints:
            logger.warning(
                "The workers of DIFFERENTIAL_EVOLUTION bring no speed-up "
                "with constraints, as SciPy evaluates the constraints "
                "in the calling process; they are not used."
            )
            workers = settings["workers"] = 1

        # The default updating strategy follows the effective workers,
        # i.e. after they are dropped because of the constraints.
        if settings["updating"] is None:
            settings["updating"] = (
                DifferentialEvolutionUpdating.IMMEDIATE
                if workers == 1
                else DifferentialEvolutionUpdating.DEFERRED
            )

        # Deactivate termination criteria which are handled by GEMSEO
        settings["maxiter"] = maxsize
        return self.__get_bounds(problem)

    def __prepare_dual_annealing(
        self, problem: OptimizationProblem, settings: MutableMapping[str, Any]
    ) -> list[tuple[float | None, float | None]]:
        """Prepare the settings of dual annealing.

        Args:
            problem: The problem to be solved.
            settings: The settings to update in place.

        Returns:
            The bounds.
        """
        # Deactivate termination criteria which are handled by GEMSEO
        settings["maxiter"] = settings["maxfun"] = maxsize
        return self.__get_bounds(problem)

    def __prepare_direct(
        self, problem: OptimizationProblem, settings: MutableMapping[str, Any]
    ) -> list[tuple[float | None, float | None]]:
        """Prepare the settings of DIRECT.

        Args:
            problem: The problem to be solved.
            settings: The settings to update in place.

        Returns:
            The bounds.
        """
        # Deactivate termination criteria which are handled by GEMSEO
        settings["maxfun"] = settings["maxiter"] = self._settings.max_iter
        return self.__get_bounds(problem)

    def __prepare_basinhopping(
        self, problem: OptimizationProblem, settings: MutableMapping[str, Any]
    ) -> NDArray[float64]:
        """Replace the GEMSEO-only settings of basin-hopping with SciPy ones.

        Args:
            problem: The problem to be solved.
            settings: The settings to update in place.

        Returns:
            The initial point.
            Basin-hopping has no `bounds` argument:
            the bounds go to the local method.
        """
        x_0, lower_bound, upper_bound = get_value_and_bounds(problem.input_space)
        settings.pop("local_method")
        local_method = self.__basinhopping_local_method
        minimizer_kwargs: dict[str, Any] = {
            "method": local_method.value,
            "bounds": Bounds(lower_bound, upper_bound),
            "options": dict(settings.pop("local_options")),
        }
        if local_method in self.__local_methods_with_gradient:
            minimizer_kwargs["jac"] = problem.objective.jac

        if problem.constraints:
            minimizer_kwargs["constraints"] = (
                self.__get_constraints_as_scipy_dictionary(problem)
            )

        settings["T"] = settings.pop("temperature")
        # GEMSEO stops the driver itself.
        settings["niter"] = maxsize
        settings["minimizer_kwargs"] = minimizer_kwargs
        # A single generator drives both the steps and the Metropolis test,
        # passed with the `rng` argument of SciPy,
        # which supersedes its legacy `seed` argument.
        rng = default_rng(settings.pop("seed"))
        settings["rng"] = rng
        settings["take_step"] = _BoundedRandomDisplacement(
            settings["stepsize"], rng, lower_bound, upper_bound
        )
        return x_0

    @staticmethod
    def __get_non_linear_constraints(
        problem: OptimizationProblem,
    ) -> tuple[NonlinearConstraint, ...]:
        """Return the SciPy nonlinear constraints.

        Args:
            problem: The problem to be solved.

        Returns:
            The SciPy nonlinear constraints.
        """
        eq_tolerance = problem.tolerances.equality
        constraints = [
            NonlinearConstraint(
                constr.evaluate, -eq_tolerance, eq_tolerance, jac=constr.jac
            )
            for constr in problem.constraints.get_equality_constraints()
        ]
        ineq_tolerance = problem.tolerances.inequality
        constraints.extend([
            NonlinearConstraint(
                constr.evaluate, -np_inf, ineq_tolerance, jac=constr.jac
            )
            for constr in problem.constraints.get_inequality_constraints()
        ])
        return tuple(constraints)

    def __get_constraints_as_scipy_dictionary(
        self, problem: OptimizationProblem
    ) -> list[dict[str, str | WrappedFunctionType | WrappedJacobianType]]:
        """Create the constraints to be passed to a SciPy algorithm as dictionaries.

        Args:
            problem: The problem to be solved.

        Returns:
            The constraints.
        """
        return [
            {
                "type": constraint.f_type,
                "fun": constraint.evaluate,
                "jac": constraint.jac,
            }
            for constraint in self._get_right_sign_constraints(problem)
        ]
