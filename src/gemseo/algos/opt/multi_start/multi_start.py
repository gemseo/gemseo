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

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.multi_start._multi_start_settings import MultiStartSettings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.utils.multiprocessing.execution import execute

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.algos.base_driver_library import DriverLibrarySettingType
    from gemseo.algos.optimization_result import OptimizationResult
    from gemseo.typing import RealArray


class MultiStart(BaseOptimizationLibrary):
    """Multi-start optimization."""

    ALGORITHM_INFOS: ClassVar[dict[str, OptimizationAlgorithmDescription]] = {
        "MultiStart": OptimizationAlgorithmDescription(
            "Multi-start optimization",
            "MultiStart",
            description=(
                "The optimization algorithm ``multistart`` "
                "generates starting points using a DOE algorithm"
                "and run a sub-optimization algorithm from each starting point."
                "Depending on the sub-optimization algorithm,"
                "``multistart`` can handle integer design variables,"
                "equality and inequality constraints"
                "as well as multi-objective functions."
            ),
            handle_multiobjective=True,
            handle_integer_variables=True,
            handle_equality_constraints=True,
            handle_inequality_constraints=True,
            Settings=MultiStartSettings,
        )
    }

    __opt_algo_settings: Mapping[str, DriverLibrarySettingType]
    """The settings of the sub-optimization algorithm."""

    def __init__(self, algo_name: str = "MultiStart") -> None:  # noqa: D107
        super().__init__(algo_name)

    def _run(self, problem: OptimizationProblem, **settings: Any) -> OptimizationResult:
        """
        Raises:
            ValueError: When ``max_iter``, ``n_start`` and ``opt_algo_max_iter``
                are not consistent.
        """  # noqa: D205 D212
        design_space = problem.design_space
        # We decrement the maximum number of iterations by one
        # as a first iteration has already been done in OptimizationLibrary._pre_run.
        max_iter = settings["max_iter"] - 1
        n_processes = settings["n_processes"]
        n_start = settings["n_start"]
        opt_algo_max_iter = settings["opt_algo_max_iter"]
        opt_algo_options = settings["opt_algo_options"]

        if opt_algo_max_iter == 0:
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

        else:
            opt_algo_max_iter = [opt_algo_max_iter] * n_start

        sum_max_iter = sum(opt_algo_max_iter)
        if sum_max_iter > max_iter:
            msg = (
                "Multi-start optimization: "
                f"the sum of the maximum number of iterations ({sum_max_iter}) "
                f"related to the sub-optimizations "
                f"is greater than the limit ({max_iter + 1}-1={max_iter})."
            )
            raise ValueError(msg)

        self.__opt_algo_settings = settings
        opt_algo_options[self._EQ_TOLERANCE] = self.problem.tolerances.equality
        opt_algo_options[self._INEQ_TOLERANCE] = self.problem.tolerances.inequality
        opt_algo_options[self._STOP_CRIT_NX] = self._f_tol_tester.n_last_iterations
        opt_algo_options[self._F_TOL_ABS] = self._f_tol_tester.absolute
        opt_algo_options[self._F_TOL_REL] = self._f_tol_tester.relative
        opt_algo_options[self._X_TOL_ABS] = self._x_tol_tester.absolute
        opt_algo_options[self._X_TOL_REL] = self._x_tol_tester.relative
        opt_algo_options["log_problem"] = False
        opt_algo_options[self._ACTIVATE_PROGRESS_BAR] = False

        doe_algo = DOELibraryFactory().create(settings["doe_algo_name"])
        if (
            "n_samples"
            in doe_algo.ALGORITHM_INFOS[settings["doe_algo_name"]].Settings.model_fields
        ):
            settings["doe_algo_options"]["n_samples"] = n_start
        samples = doe_algo.compute_doe(design_space, **settings["doe_algo_options"])

        problems = execute(
            self._optimize, (), n_processes, list(zip(samples, opt_algo_max_iter))
        )

        # The sub-optimizations use their own optimization problems
        # and so their own databases.
        # When the sub-optimizations are run sequentially,
        # the evaluations are stored both in the main database and in the sub-databases.
        # When the sub-optimizations are run in parallel,
        # the evaluations are stored in the sub-databases only,
        # and we need to store them manually in the main database.
        if n_processes > 1:
            for problem in problems:
                database = problem.database
                f_hist, x_hist = database.get_function_history(
                    self.problem.objective.name, with_x_vect=True
                )
                for xi, fi in zip(x_hist, f_hist):
                    self.problem.database.store(xi, {self.problem.objective.name: fi})

                for functions in [self.problem.constraints, self.problem.observables]:
                    for f in functions:
                        f_hist, x_hist = database.get_function_history(
                            f.name, with_x_vect=True
                        )
                        for xi, fi in zip(x_hist, f_hist):
                            self.problem.database.store(xi, {f.name: fi})

        file_path = settings["multistart_file_path"]
        if file_path:
            problem = OptimizationProblem(design_space)
            problem.objective = self.problem.objective
            problem.constraints = self.problem.constraints
            for sub_problem in problems:
                x_opt = self._get_optimum_from_database(sub_problem).x_opt
                problem.database.store(x_opt, sub_problem.database[x_opt])
            problem.to_hdf(file_path)

        return self._get_optimum_from_database(problem)

    def _optimize(self, data: tuple[RealArray, int]) -> OptimizationProblem:
        """Solve the sub-optimization problem from an initial design value.

        Args:
            data: The initial design value and the maximum number of iterations.

        Returns:
            The sub-optimization problem.
        """
        initial_point, max_iter = data

        design_space = deepcopy(self.problem.design_space)
        design_space.set_current_value(initial_point)

        problem = OptimizationProblem(design_space)
        problem.objective = self.problem.objective
        problem.constraints = self.problem.constraints
        problem.observables = self.problem.observables

        factory = OptimizationLibraryFactory()
        opt_algo = factory.create(self.__opt_algo_settings["opt_algo_name"])
        opt_algo.execute(
            problem,
            max_iter=max_iter,
            **self.__opt_algo_settings["opt_algo_options"],
        )
        return problem
