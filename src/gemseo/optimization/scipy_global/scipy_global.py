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

from numpy import float64
from numpy import inf as np_inf
from numpy import int32
from numpy import isfinite
from numpy import real
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint
from scipy.optimize import differential_evolution
from scipy.optimize import dual_annealing
from scipy.optimize import shgo

from gemseo.optimization.core.base_optimization_library import BaseOptimizationLibrary
from gemseo.optimization.core.base_optimization_library import (
    OptimizationAlgorithmDescription,
)
from gemseo.optimization.scipy_global.settings.base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.optimization.scipy_global.settings.differential_evolution import (
    DIFFERENTIAL_EVOLUTION_Settings,
)
from gemseo.optimization.scipy_global.settings.differential_evolution import (
    DifferentialEvolutionUpdating,
)
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
    from concurrent.futures import ProcessPoolExecutor

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
        "DIFFERENTIAL_EVOLUTION": differential_evolution,
        "DUAL_ANNEALING": dual_annealing,
        "SHGO": shgo,
    })
    """The mapping between the algorithm name and the SciPy function."""

    __doc: Final[str] = "https://docs.scipy.org/doc/scipy/reference/generated/"

    ALGORITHM_INFOS: ClassVar[dict[str, SciPyGlobalAlgorithmDescription]] = {
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
        # The problem works in the coordinates the algorithm manipulates,
        # so its space already carries the bounds the algorithm needs.
        _, l_b, u_b = get_value_and_bounds(problem.input_space)
        # Replace infinite values with None:
        l_b = [val if isfinite(val) else None for val in l_b]
        u_b = [val if isfinite(val) else None for val in u_b]
        bounds = list(zip(l_b, u_b, strict=False))

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
            if self._algo_name == "SHGO":
                constraints = self.__get_constraints_as_scipy_dictionary(problem)
                filtered_settings["constraints"] = constraints
            elif self._algo_name == "DIFFERENTIAL_EVOLUTION":
                constraints = self.__get_non_linear_constraints(problem)
                filtered_settings["constraints"] = constraints
                # The algorithm handles integer variables itself,
                # so the working space still carries their (integral) bounds,
                # as `DesignSpace` normalization leaves integer variables alone;
                # SciPy is told which components are integer so that it keeps
                # them integral at every evaluation instead of exploring them
                # continuously, unlike the discrete ones, which stay unhandled.
                filtered_settings["integrality"] = (
                    problem.input_space.get_integer_mask()
                )

            workers = filtered_settings.get("workers", 1)
            if (
                workers != 1
                and self._algo_name == "DIFFERENTIAL_EVOLUTION"
                and problem.constraints
            ):
                logger.warning(
                    "The workers of DIFFERENTIAL_EVOLUTION bring no speed-up "
                    "with constraints, as SciPy evaluates the constraints "
                    "in the calling process; they are not used."
                )
                workers = filtered_settings["workers"] = 1

            # The default updating strategy follows the effective workers,
            # i.e. after they are dropped because of the constraints.
            if (
                self._algo_name == "DIFFERENTIAL_EVOLUTION"
                and filtered_settings["updating"] is None
            ):
                filtered_settings["updating"] = (
                    DifferentialEvolutionUpdating.IMMEDIATE
                    if workers == 1
                    else DifferentialEvolutionUpdating.DEFERRED
                )

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

            # Deactivate termination criteria which are handled by GEMSEO
            if self._algo_name == "SHGO":
                filtered_settings["options"].update(
                    dict.fromkeys(["maxev", "maxfev", "maxiter", "maxtime"], maxsize)
                )
                filtered_settings["options"]["ftol"] = 0.0
            elif self._algo_name == "DUAL_ANNEALING":
                filtered_settings["maxiter"] = filtered_settings["maxfun"] = maxsize
            else:  # Necessarily the differential evolution algorithm
                filtered_settings["maxiter"] = maxsize

            global_optimizer = self.__names_to_functions[self._algo_name]
            opt_result = global_optimizer(
                func=self._compute_objective,
                bounds=bounds,
                **filtered_settings,
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

        return opt_result.message, opt_result.success

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
