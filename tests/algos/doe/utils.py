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
#                      initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from copy import deepcopy
from typing import Any
from typing import Callable

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.doe_lib import DOELibrary
from gemseo.problems.analytical.rosenbrock import Rosenbrock


def get_problem(
    dim: int,
) -> Rosenbrock:
    """Return the Rosenbrock problem for the given input dimension.

    It serves as a benchmark problem.

    Args:
        dim: The dimension of the problem.

    Returns:
        The Rosenbrock problem.
    """
    problem = Rosenbrock(dim)
    problem.check()
    return problem


def execute_problem(
    doe_algo_name: str,
    dim: int = 3,
    **options: Any,
) -> DOELibrary:
    """Create and execute a problem.

    This method creates an OptimizationProblem
    with the passed input space dimension
    and evaluates its functions (objective, constraints and observables)
    with the passed algorithm name and options.

    Args:
        doe_algo_name: The name of the DOE algorithm.
        dim: The dimension of the variables space.
        options: The algorithm options.

    Returns:
        The DOE library after the execution of the DOE algorithm on the problem.
    """
    problem = get_problem(dim)
    doe_library = DOEFactory().create(doe_algo_name)
    doe_library.execute(problem, **options)
    return doe_library


def check_problem_execution(
    dim: int,
    doe_library: DOELibrary,
    algo_name: str,
    get_expected_nsamples: Callable[[str, int, int | None], int],
    options: dict[str, Any],
) -> str | None:
    """Create a problem, execute it and return an error message if any.

    Args:
        dim: The dimension of the variables space.
        doe_library: The DOE library.
        algo_name: The name of the DOE algorithm.
        get_expected_nsamples: The method returning the expected number of samples.
        options: The algorithm options.

    Returns:
        The error message, if any.
    """
    problem = get_problem(dim)
    doe_library.execute(problem, algo_name=algo_name, **options)
    samples = doe_library.unit_samples

    pb_name = problem.__class__.__name__
    error_msg = "DOE with {} failed to generate sample on problem {}".format(
        algo_name, pb_name
    )

    if not len(samples.shape) == 2 or samples.shape[0] == 0:
        error_msg += f", wrong samples shapes : {samples.shape}"
        return error_msg

    n_samples = options.get("n_samples")
    exp_samples = get_expected_nsamples(algo_name, dim, n_samples)
    get_samples = samples.shape[0]
    if exp_samples is not None and get_samples != exp_samples:
        error_msg += "\n number_samples are not the expected ones : "
        error_msg += f"\n expected : {exp_samples} got : {get_samples}"
        return error_msg


def create_test_function(
    dim: int,
    doe_library: DOELibrary,
    algo_name: str,
    get_expected_nsamples: Callable[[str, int, int | None], int],
    options: dict[str, Any],
) -> Callable[[Any | None], None]:
    """Create a test function for a DOE algorithm and a variables space dimension.

    Args:
        dim: The dimension of the variables space.
        doe_library: The DOE library.
        algo_name: The name of the DOE algorithm.
        get_expected_nsamples: The method returning the expected number of samples.
        options: The algorithm options.

    Returns:
        A function checking the execution of the problem with the given DOE algorithm
        and raising an exception if an error occurs.
    """

    def test_problem_execution() -> None:
        """Test the execution of the problem with a given DOE algorithm.

        Raises:
            Exception: The reason why the test failed.
        """
        msg = check_problem_execution(
            dim, doe_library, algo_name, get_expected_nsamples, options
        )
        if msg is not None:
            raise Exception(msg)

    return test_problem_execution


def generate_test_functions(
    opt_lib_name: str,
    get_expected_nsamples: Callable[[str, int, int | None], int],
    get_options: Callable[[str, int], dict[str, Any]],
) -> list[Callable[[], None]]:
    """Generate test functions for a DOE library.

    This methods filters the algorithms adapted to the benchmark problem.

    Args:
        opt_lib_name: The name of the library.
        get_expected_nsamples: The method returning the expected number of samples.
        get_options: The method returning the algorithm options.

    Returns:
        The test functions.
    """
    tests = []
    factory = DOEFactory()

    if factory.is_available(opt_lib_name):
        for dim in [1, 5]:
            opt_lib = DOEFactory().create(opt_lib_name)
            algos = opt_lib.filter_adapted_algorithms(get_problem(dim))
            for algo_name in algos:
                options = deepcopy(get_options(algo_name, dim))
                # Must copy options otherwise they are erased in the loop
                test_method = create_test_function(
                    dim,
                    opt_lib,
                    algo_name,
                    get_expected_nsamples,
                    deepcopy(options),
                )
                name = "test_{}_lib_{}_on_Rosenbrock_n_{}".format(
                    opt_lib.__class__.__name__, algo_name, dim
                )
                name = name.replace("-", "_")
                test_method.__name__ = name
                tests.append(test_method)
    return tests
