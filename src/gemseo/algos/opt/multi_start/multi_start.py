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
from typing import ClassVar

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.opt.multi_start.settings.multi_start_settings import (
    MultiStart_Settings,
)
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.utils.multiprocessing.execution import execute

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class MultiStart(BaseOptimizationLibrary[MultiStart_Settings]):
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
            Settings=MultiStart_Settings,
        )
    }

    def __init__(self, algo_name: str = "MultiStart") -> None:  # noqa: D107
        super().__init__(algo_name)

    def _run(self, problem: OptimizationProblem) -> None:
        design_space = problem.design_space
        # We decrement the maximum number of iterations by one
        # as a first iteration has already been done in OptimizationLibrary._pre_run.
        max_iter = self._settings.max_iter - 1
        n_processes = self._settings.n_processes
        n_start = self._settings.n_start
        opt_algo_max_iter = self._settings.opt_algo_max_iter

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

        doe_algo = DOELibraryFactory().create(self._settings.doe_algo_name)
        if (
            "n_samples"
            in doe_algo.ALGORITHM_INFOS[
                self._settings.doe_algo_name
            ].Settings.model_fields
        ):
            self._settings.doe_algo_settings["n_samples"] = n_start
        samples = doe_algo.compute_doe(design_space, **self._settings.doe_algo_settings)

        problems = execute(
            self._optimize,
            (),
            n_processes,
            list(zip(samples, opt_algo_max_iter, strict=False)),
        )

        for problem in problems:
            database = problem.database
            f_hist, x_hist = database.get_function_history(
                self._problem.objective.name, with_x_vect=True
            )
            for xi, fi in zip(x_hist, f_hist, strict=False):
                self._problem.database.store(xi, {self._problem.objective.name: fi})

            for functions in [self._problem.constraints, self._problem.observables]:
                for f in functions:
                    f_hist, x_hist = database.get_function_history(
                        f.name, with_x_vect=True
                    )
                    for xi, fi in zip(x_hist, f_hist, strict=False):
                        self._problem.database.store(xi, {f.name: fi})

        file_path = self._settings.multistart_file_path
        if file_path:
            problem = OptimizationProblem(design_space)
            problem.objective = self._problem.objective
            problem.constraints = self._problem.constraints
            for sub_problem in problems:
                x_opt = self._get_result(sub_problem, None, None).x_opt
                problem.database.store(x_opt, sub_problem.database[x_opt])
            problem.to_hdf(file_path)

    def _optimize(self, data: tuple[RealArray, int]) -> OptimizationProblem:
        """Solve the sub-optimization problem from an initial design value.

        Args:
            data: The initial design value and the maximum number of iterations.

        Returns:
            The sub-optimization problem.
        """
        initial_point, max_iter = data

        design_space = deepcopy(self._problem.design_space)
        design_space.set_current_value(initial_point)

        problem = OptimizationProblem(design_space)
        problem.differentiation_method = self._problem.differentiation_method
        problem.objective = self._problem.objective.original
        problem.constraints = (c.original for c in self._problem.constraints)
        problem.observables = (o.original for o in self._problem.observables)

        factory = OptimizationLibraryFactory()
        opt_algo = factory.create(self._settings.opt_algo_name)
        opt_algo.execute(
            problem,
            max_iter=max_iter,
            **self._settings.opt_algo_settings,
        )
        return problem
