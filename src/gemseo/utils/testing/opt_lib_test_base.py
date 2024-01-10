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

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.rastrigin import Rastrigin
from gemseo.problems.analytical.rosenbrock import Rosenbrock

if TYPE_CHECKING:
    from numpy._typing import NDArray

    from gemseo.algos.opt.optimization_library import OptimizationLibrary
    from gemseo.algos.opt_problem import OptimizationProblem


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
        opt_lib_name: str, algo_name: str, **options: Any
    ) -> OptimizationLibrary:
        """Solve the Power 2 problem with an optimization library.

        This optimization problem has equality constraints.

        Args:
            opt_lib_name: The name of the optimization library.
            algo_name: The name of the optimization algorithm.
            **options: The options of the optimization algorithm.

        Returns:
            An optimization library after the resolution of the Power 2 problem.
        """
        problem = OptLibraryTestBase().get_pb_instance("Power2")
        opt_library = OptimizersFactory().create(opt_lib_name)
        opt_library.execute(problem, algo_name=algo_name, **options)
        return opt_library

    @staticmethod
    def generate_one_test_unconstrained(opt_lib_name, algo_name, **options):
        """Solve the Rosenbrock problem with an optimization library.

        This optimization problem has no constraints.

        Args:
            opt_lib_name: The name of the optimization library.
            algo_name: The name of the optimization algorithm.
            **options: The options of the optimization algorithm.

        Returns:
            An optimization library after the resolution of the Rosenbrock problem.
        """
        problem = OptLibraryTestBase().get_pb_instance("Rosenbrock")
        opt_library = OptimizersFactory().create(opt_lib_name)
        opt_library.execute(problem, algo_name=algo_name, **options)
        return opt_library

    @staticmethod
    def generate_error_test(opt_lib_name, algo_name, **options):
        """Solve the Power 2 problem with an optimization library.

        This optimization problem has constraints.

        This problem raises an error when calling the objective.

        Args:
            opt_lib_name: The name of the optimization library.
            algo_name: The name of the optimization algorithm.
            **options: The options of the optimization algorithm.

        Returns:
            An optimization library after the resolution of the Rosenbrock problem.
        """
        problem = Power2(exception_error=True)
        opt_library = OptimizersFactory().create(opt_lib_name)
        opt_library.execute(problem, algo_name=algo_name, **options)
        return opt_library

    @staticmethod
    def run_and_test_problem(
        problem: OptimizationProblem, opt_library: str, algo_name: str, **options: Any
    ) -> str | None:
        """Run and test an optimization algorithm.

        Args:
            problem: The optimization problem.
            opt_library: The name of the optimization library.
            algo_name: The name of the optimization algorithm.
            **options: The options of the optimization algorithm.

        Returns:
            The error message if the optimizer cannot find the solution,
            otherwise ``None``.
        """
        opt = opt_library.execute(problem, algo_name=algo_name, **options)
        x_opt, f_opt = problem.get_solution()
        x_err = OptLibraryTestBase.relative_norm(opt.x_opt, x_opt)
        f_err = OptLibraryTestBase.relative_norm(opt.f_opt, f_opt)

        if x_err > 1e-2 or f_err > 1e-2:
            pb_name = problem.__class__.__name__
            return (
                "Optimization with "
                + algo_name
                + " failed to find solution of problem "
                + pb_name
                + " after n calls = "
                + str(len(problem.database))
            )
        return None

    @staticmethod
    def create_test(
        problem: OptimizationProblem,
        opt_library: str,
        algo_name: str,
        options: dict[str, Any],
    ):
        """Create a function to run and test an optimization algorithm.

        Args:
            problem: The optimization problem.
            opt_library: The name of the optimization library.
            algo_name: The name of the optimization algorithm.
            options: The options of the optimization algorithm.

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
                problem, opt_library, algo_name, **options
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

    def generate_test(self, opt_lib_name, get_options=None, get_problem_options=None):
        """Generates the tests for an opt library Filters algorithms adapted to the
        benchmark problems.

        Args:
            opt_lib_name: The name of the optimization library.
            get_options: A function to get the options of the algorithm.
            get_problem_options: A function to get the options of the problem.

        Returns!
            The test methods to be attached to a unitest class.
        """
        tests = []
        factory = OptimizersFactory()
        if factory.is_available(opt_lib_name):
            opt_lib = OptimizersFactory().create(opt_lib_name)
            for pb_name in ["Rosenbrock", "Power2", "Rastrigin"]:
                if get_problem_options is not None:
                    pb_options = get_problem_options(pb_name)
                else:
                    pb_options = {}
                problem = self.get_pb_instance(pb_name, pb_options)
                algos = opt_lib.filter_adapted_algorithms(problem)
                for algo_name in algos:
                    # Reinitialize problem between runs
                    problem = self.get_pb_instance(pb_name, pb_options)
                    if get_options is not None:
                        options = get_options(algo_name)
                    else:
                        options = {"max_iter": 10000}
                    test_method = self.create_test(problem, opt_lib, algo_name, options)
                    name = "test_" + opt_lib.__class__.__name__ + "_" + algo_name
                    name += "_on_" + problem.__class__.__name__
                    name = name.replace("-", "_")
                    test_method.__name__ = name

                    tests.append(test_method)
        return tests
