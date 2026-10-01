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
"""Multi-start optimization."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.core.problem.database import Database
from gemseo.doe.factory import doe_library_factory
from gemseo.optimization.core.base_optimization_library import BaseOptimizationLibrary
from gemseo.optimization.core.base_optimization_library import (
    OptimizationAlgorithmDescription,
)
from gemseo.optimization.multi_start.settings.multi_start_settings import (
    MultiStart_Settings,
)
from gemseo.optimization.problem import OptimizationProblem
from gemseo.util.multiprocessing.execution import execute

if TYPE_CHECKING:
    from gemseo.core.problem.database import DatabaseValueType
    from gemseo.core.problem.database import FunctionOutputValueType
    from gemseo.util.typing import RealArray

logger = logging.getLogger(__name__)


class MultiStart(BaseOptimizationLibrary[MultiStart_Settings]):
    """Multi-start optimization."""

    ALGORITHM_INFOS: ClassVar[dict[str, OptimizationAlgorithmDescription]] = {
        "MultiStart": OptimizationAlgorithmDescription(
            "Multi-start optimization",
            "MultiStart",
            description=(
                "The optimization algorithm `multistart` "
                "generates starting points using a DOE algorithm"
                "and run a sub-optimization algorithm from each starting point."
                "Depending on the sub-optimization algorithm,"
                "`multistart` can handle integer design variables,"
                "equality and inequality constraints"
                "as well as multi-objective functions."
            ),
            handle_multiobjective=True,
            handle_discrete_variables=True,
            handle_integer_variables=True,
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            settings_class=MultiStart_Settings,
        )
    }

    _iterates_on_working_problem: ClassVar[bool] = False
    """Multi-start builds sub-problems and hands them to sub-drivers.

    Each sub-problem pairs a copy of the input space with the original functions,
    so both have to be the ones the user declared.
    """

    def __init__(self, algo_name: str = "MultiStart") -> None:  # noqa: D107
        super().__init__(algo_name)

    def _run(self, problem: OptimizationProblem) -> None:
        design_space = problem.input_space
        # We decrement the maximum number of iterations by one
        # as a first iteration has already been done in OptimizationLibrary._pre_run.
        max_iter = self._settings.max_iter - 1
        n_processes = self._settings.n_processes

        # The starting points correspond to the input samples of the DOE algorithm.
        if "samples" in self._settings.doe_algo_settings.model_fields:
            # The input samples of the DOE algorithm are defined in the settings,
            # through the `samples` field, e.g. CustomDOE algorithm.
            n_start = len(self._settings.doe_algo_settings.samples)
        else:
            # The input samples of the DOE algorithm are defined in the settings,
            # through the `n_samples` field, e.g. MC_Settings.
            n_start = self._settings.doe_algo_settings.n_samples

        # We define the maximum number of iterations of the sub-optimization algorithms.
        if "max_iter" in self._settings.opt_algo_settings.model_fields_set:
            # The user sets this number in the setting of these algorithms.
            opt_algo_max_iter = [self._settings.opt_algo_settings.max_iter] * n_start
            sum_max_iter = sum(opt_algo_max_iter)
            if sum_max_iter > max_iter:
                msg = (
                    "Multi-start optimization: "
                    f"the sum of the maximum number of iterations ({sum_max_iter}) "
                    f"related to the sub-optimizations "
                    f"is greater than the limit ({max_iter + 1}-1={max_iter})."
                )
                raise ValueError(msg)
        else:
            # This number is deduced from the total maximum number of iterations
            # defined by the `max_iter` option.
            # This total number is allocated fairly
            # among the different sub-optimizations.
            if max_iter < n_start:
                msg = (
                    "Multi-start optimization: "
                    f"the maximum number of iterations ({max_iter + 1}) "
                    f"must be greater than the number of initial points ({n_start})."
                )
                raise ValueError(msg)

            n = int(max_iter / n_start)
            opt_algo_max_iter = [n] * n_start
            for i in range(max_iter - n * n_start):
                opt_algo_max_iter[i] += 1

        doe_algo = doe_library_factory.create(
            self._settings.doe_algo_settings.target_class_name
        )
        samples = doe_algo.sample_space(
            design_space, settings=self._settings.doe_algo_settings
        )

        skip_failed = self._settings.skip_failed_starting_points
        store_jacobian = self._settings.store_jacobian
        # Re-raise the errors of the sub-optimizations when they are not
        # skipped, so that the parallel mode surfaces the same error as the
        # serial mode.
        exceptions_to_re_raise = () if skip_failed else (Exception,)
        results = execute(
            self._optimize,
            (),
            n_processes,
            list(zip(samples, opt_algo_max_iter, strict=False)),
            exceptions_to_re_raise=exceptions_to_re_raise,
        )
        objective_name = self._problem.objective.name
        sub_problems = []
        # The reason of the last failure, reported if all the starting points fail.
        # It is a message and not an exception
        # because the workers of the parallel mode return messages.
        last_error = ""
        for index, (starting_point, result) in enumerate(
            zip(samples, results, strict=False)
        ):
            if skip_failed:
                if result is None:
                    sub_problem, error = None, "no sub-optimization problem"
                else:
                    sub_problem, error = result

                if sub_problem is None:
                    # There is no partial history to merge.
                    has_objective = False
                else:
                    has_objective = (
                        objective_name in sub_problem.database.get_function_names()
                    )
                    if not error and not has_objective:
                        error = f"no evaluation of the objective {objective_name!r}"

                if error:
                    last_error = error
                    logger.warning(
                        "Multi-start optimization: "
                        "skipping the starting point %s (%s) "
                        "because the sub-optimization failed (%s).",
                        index + 1,
                        starting_point,
                        error,
                    )
                    if not has_objective:
                        # Nothing was evaluated before the failure: skip entirely.
                        continue
                    # The sub-optimization failed but evaluated the objective
                    # at least once: merge its entries below,
                    # without counting it as a survivor.
                else:
                    sub_problems.append(sub_problem)
            else:
                sub_problem, _ = result
                sub_problems.append(sub_problem)

                if objective_name not in sub_problem.database.get_function_names():
                    msg = (
                        "Multi-start optimization: "
                        "the sub-optimization from the starting point "
                        f"{index + 1} ({starting_point}) evaluated no value "
                        f"of the objective {objective_name!r}; "
                        "set skip_failed_starting_points back to its default value "
                        "True to skip it."
                    )
                    raise ValueError(msg)

            for x_vect, outputs in sub_problem.database.items():
                self._problem.database.store(
                    x_vect.wrapped_array,
                    self.__copy_outputs(outputs, store_jacobian),
                )

            self._problem.database.relaxed_variable_names.update(
                sub_problem.database.relaxed_variable_names
                & set(self._problem.database.input_space)
            )

        if skip_failed and not sub_problems:
            msg = (
                "Multi-start optimization: "
                f"all the {len(results)} starting points failed; "
                f"the last one failed with: {last_error}"
            )
            raise ValueError(msg)

        file_path = self._settings.multistart_file_path
        if file_path:
            local_optima_problem = OptimizationProblem(design_space)
            local_optima_problem.objective = self._problem.objective
            local_optima_problem.constraints = self._problem.constraints
            for sub_problem in sub_problems:
                x_opt = self._get_result(sub_problem, None, None).x_opt
                outputs = sub_problem.database[x_opt]
                local_optima_problem.database.store(
                    x_opt, self.__copy_outputs(outputs, store_jacobian)
                )
            local_optima_problem.to_hdf(file_path)

    @staticmethod
    def __copy_outputs(
        outputs: DatabaseValueType, store_jacobian: bool
    ) -> dict[str, FunctionOutputValueType]:
        """Copy the output values of a database entry, optionally dropping Jacobians.

        Args:
            outputs: The output values of a database entry,
                possibly including Jacobian entries
                whose names start with `Database.GRAD_TAG`.
            store_jacobian: Whether to keep the Jacobian entries in the copy.

        Returns:
            A new mapping with the output values,
            excluding the Jacobian entries when `store_jacobian` is `False`.
        """
        if store_jacobian:
            return dict(outputs)

        return {
            name: value
            for name, value in outputs.items()
            if not name.startswith(Database.GRAD_TAG)
        }

    def _optimize(self, data: tuple[RealArray, int]) -> tuple[OptimizationProblem, str]:
        """Solve the sub-optimization problem from an initial design value.

        Args:
            data: The initial design value and the maximum number of iterations.

        Returns:
            The sub-optimization problem,
            and the reason why the sub-optimization failed,
            or an empty string in the case of success.
        """
        initial_point, max_iter = data

        design_space = deepcopy(self._problem.input_space)
        design_space.set_current_value(initial_point)

        problem = OptimizationProblem(design_space)
        problem.differentiation_method = self._problem.differentiation_method
        problem.objective = self._problem.objective.original
        problem.constraints = (c.original for c in self._problem.constraints)
        problem.observables = (o.original for o in self._problem.observables)

        # Imported here because instantiating this factory imports this module.
        from gemseo.optimization.factory import optimization_library_factory

        self._settings.opt_algo_settings.max_iter = max_iter
        try:
            optimization_library_factory.execute(
                problem, settings=self._settings.opt_algo_settings
            )
        except Exception as error:  # noqa: BLE001
            if not self._settings.skip_failed_starting_points:
                raise
            return problem, f"{type(error).__name__}: {error}"

        return problem, ""
