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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Common tools for testing opt libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.problems.optimization.power_2 import Power2
from gemseo.problems.optimization.rastrigin import Rastrigin
from gemseo.problems.optimization.rosenbrock import Rosenbrock

if TYPE_CHECKING:
    from numpy._typing import NDArray

    from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
    from gemseo.algos.optimization_problem import OptimizationProblem


class OptLibraryTestBase:
    """Main testing class."""

    @staticmethod
    def relative_norm(x: NDArray[float], x_ref: NDArray[float]) -> float:
        """Compute a relative Euclidean norm between to vectors.

        Args:
            x: The vector.
            x_ref: The reference vector.

        Returns:
            The relative Euclidean norm between two vectors.
        """
        xr_norm = np.linalg.norm(x_ref)
        if xr_norm < 1e-8:
            return np.linalg.norm(x - x_ref)
        return np.linalg.norm(x - x_ref) / xr_norm

    @staticmethod
    def norm(x: NDArray[float]) -> float:
        """Compute the Euclidean norm of a vector.

        Args:
            x: The vector.

        Returns:
            The Euclidean norm of the vector.
        """
        return np.linalg.norm(x)

    @staticmethod
    def generate_one_test(
        algo_name: str, **settings: Any
    ) -> tuple[BaseOptimizationLibrary, OptimizationProblem]:
        """Solve the Power 2 problem with an optimization library.

        This optimization problem has equality constraints.

        Args:
            algo_name: The name of the optimization algorithm.
            **settings: The settings of the optimization algorithm.

        Returns:
            An optimization library after the resolution of the Power 2 problem.
        """
        problem = OptLibraryTestBase().get_pb_instance("Power2")
        opt_library = OptimizationLibraryFactory().create(algo_name)
        opt_library.execute(problem, **settings)
        return opt_library, problem

    @staticmethod
    def generate_one_test_unconstrained(opt_lib_name, algo_name, **settings):
        """Solve the Rosenbrock problem with an optimization library.

        This optimization problem has no constraints.

        Args:
            opt_lib_name: The name of the optimization library.
            algo_name: The name of the optimization algorithm.
            **settings: The settings of the optimization algorithm.

        Returns:
            An optimization library after the resolution of the Rosenbrock problem.
        """
        problem = OptLibraryTestBase().get_pb_instance("Rosenbrock")
        opt_library = OptimizationLibraryFactory().create(algo_name)
        opt_library.execute(problem, **settings)
        return problem

    @staticmethod
    def generate_error_test(opt_lib_name, algo_name, **settings):
        """Solve the Power 2 problem with an optimization library.

        This optimization problem has constraints.

        This problem raises an error when calling the objective.

        Args:
            opt_lib_name: The name of the optimization library.
            algo_name: The name of the optimization algorithm.
            **settings: The settings of the optimization algorithm.

        Returns:
            An optimization library after the resolution of the Rosenbrock problem.
        """
        problem = Power2(exception_error=True)
        opt_library = OptimizationLibraryFactory().create(algo_name)
        opt_library.execute(problem, **settings)
        return opt_library

    @staticmethod
    def run_and_test_problem(
        problem: OptimizationProblem, algo_name: str, **settings: Any
    ) -> str | None:
        """Run and test an optimization algorithm.

        Args:
            problem: The optimization problem.
            algo_name: The name of the optimization algorithm.
            **settings: The settings of the optimization algorithm.

        Returns:
            The error message if the optimizer cannot find the solution,
            otherwise ``None``.
        """
        opt = OptimizationLibraryFactory().execute(
            problem, algo_name=algo_name, **settings
        )
        x_opt, f_opt = problem.get_solution()
        x_err = OptLibraryTestBase.relative_norm(opt.x_opt, x_opt)
        f_err = OptLibraryTestBase.relative_norm(opt.f_opt, f_opt)

        if x_err > 1e-2 or f_err > 1e-2:
            pb_name = problem.__class__.__name__
            return (
                f"Optimization with {algo_name} failed "
                f"to find solution of problem {pb_name} "
                f"after n calls = {len(problem.database)}"
            )
        return None

    @staticmethod
    def create_test(
        problem: OptimizationProblem,
        algo_name: str,
        settings: dict[str, Any],
    ):
        """Create a function to run and test an optimization algorithm.

        Args:
            problem: The optimization problem.
            algo_name: The name of the optimization algorithm.
            settings: The settings of the optimization algorithm.

        Returns:
            The error message if the optimizer cannot find the solution,
            otherwise ``None``.
        """

        def test_algo(self=None) -> None:
            """Test the algorithm.

            Raises:
                RuntimeError: When the algorithm cannot find the solution.
            """
            msg = OptLibraryTestBase.run_and_test_problem(
                problem, algo_name, **settings
            )
            if msg is not None:
                raise RuntimeError(msg)
            return msg

        return test_algo

    @staticmethod
    def get_pb_instance(
        pb_name: str,
        pb_options: dict[str, Any] | None = None,
    ) -> Rosenbrock | Power2 | Rastrigin:
        """Return an optimization problem.

        Args:
            pb_name: The name of the optimization problem.
            pb_options: The options to be passed to the optimization problem.

        Raises:
            ValueError: When the problem is not available.
        """
        if pb_options is None:
            pb_options = {}
        if pb_name == "Rosenbrock":
            return Rosenbrock(2, **pb_options)
        if pb_name == "Power2":
            return Power2(**pb_options)
        if pb_name == "Rastrigin":
            return Rastrigin(**pb_options)
        msg = f"Bad pb_name argument: {pb_name}"
        raise ValueError(msg)

    def generate_test(self, opt_lib_name, get_settings=None, get_problem_options=None):
        """Generates the tests for an opt library Filters algorithms adapted to the
        benchmark problems.

        Args:
            opt_lib_name: The name of the optimization library.
            get_settings: A function to get the settings of the algorithm.
            get_problem_options: A function to get the options of the problem.

        Returns:
            The test methods to be attached to an unitest class.
        """
        tests = []
        factory = OptimizationLibraryFactory()
        if factory.is_available(opt_lib_name):
            cls = OptimizationLibraryFactory().get_class(opt_lib_name)
            for pb_name in ["Rosenbrock", "Power2", "Rastrigin"]:
                if get_problem_options is not None:
                    pb_options = get_problem_options(pb_name)
                else:
                    pb_options = {}
                problem = self.get_pb_instance(pb_name, pb_options)
                algos = cls.filter_adapted_algorithms(problem)
                for algo_name in algos:
                    # Reinitialize problem between runs
                    problem = self.get_pb_instance(pb_name, pb_options)
                    if get_settings is not None:
                        settings = get_settings(algo_name)
                    else:
                        settings = {"max_iter": 10000}
                    test_method = self.create_test(problem, algo_name, settings)
                    name = f"test_{opt_lib_name}_{algo_name}_on_{problem.__class__.__name__}"  # noqa: E501
                    name = name.replace("-", "_")
                    test_method.__name__ = name

                    tests.append(test_method)
        return tests
