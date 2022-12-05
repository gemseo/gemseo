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
#      :author: Damien Guenot - 20 avr. 2016
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re
from typing import Any
from unittest import mock

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from numpy import unique

from .utils import execute_problem
from .utils import generate_test_functions
from .utils import get_problem

DOE_LIB_NAME = "OpenTURNS"


@pytest.fixture
def identity_problem() -> OptimizationProblem:
    """A problem whose objective is the identity function defined over [0,1]."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "f")
    return problem


def test_library_from_factory():
    """Check that the DOEFactory can create the OpenTURNS library."""
    factory = DOEFactory()
    if factory.is_available(DOE_LIB_NAME):
        factory.create(DOE_LIB_NAME)


@pytest.mark.parametrize(
    "levels,msg",
    [
        (4, "Invalid options for algorithm OT_COMPOSITE"),
        ([0.1, 0.2, 1.3], "Levels must belong to [0, 1]; got [0.1, 1.3]."),
        ([-0.1, 0.2, 1.3], "Invalid options for algorithm OT_COMPOSITE"),
    ],
)
def test_composite_malformed_levels(levels, msg):
    """Check that passing malformed 'levels' raises an exception for Composite DOE."""
    with pytest.raises(ValueError, match=re.escape(msg)):
        execute_problem(
            DOE_LIB_NAME,
            algo_name="OT_COMPOSITE",
            n_samples=20,
            levels=levels,
            centers=[0.0, 0.0, 0.0],
        )


def test_malformed_levels_with_check_and_cast_levels():
    """Check that checking and casting malformed 'levels' raises an error."""
    lib = DOEFactory().create(DOE_LIB_NAME)

    with pytest.raises(
        TypeError,
        match=re.escape(
            "The argument 'levels' must be either a list or a tuple; got a 'int'."
        ),
    ):
        lib._OpenTURNS__check_and_cast_levels({"levels": 1})


@pytest.mark.parametrize(
    "centers,exception", [([0.5] * (3 + 1), ValueError), (0.5, TypeError)]
)
def test_composite_malformed_centers(centers, exception):
    """Check that passing malformed 'centers' raises an exception for Composite DOE."""
    with pytest.raises(exception):
        execute_problem(
            DOE_LIB_NAME,
            algo_name="OT_COMPOSITE",
            centers=centers,
            levels=[0.1, 0.2, 0.3],
        )


def test_malformed_centers_with_check_and_cast_centers():
    """Check that checking and casting malformed 'centers' raises an error."""
    lib = DOEFactory().create("OT_LHS")
    lib.problem = Rosenbrock()

    with pytest.raises(TypeError):
        lib._OpenTURNS__check_and_cast_centers({"centers": 1})


def test_check_stratified_options():
    """Verify that OpenTURNS.__check_stratified_options works correctly."""
    factory = DOEFactory()
    lib = factory.create(DOE_LIB_NAME)
    lib.problem = Rosenbrock()
    options = {}
    dimension = lib.problem.dimension
    with pytest.raises(
        KeyError,
        match=re.escape(
            "Missing parameter 'levels', "
            "tuple of normalized levels in [0,1] you need in your design."
        ),
    ):
        lib._OpenTURNS__check_stratified_options(dimension, options)

    options = {"levels": [0.5, 0.2]}
    lib._OpenTURNS__check_stratified_options(dimension, options)
    assert lib.CENTER_KEYWORD in options


def test_call():
    """Check that calling OpenTURNS library returns a NumPy array correctly shaped.

    Use an algorithm with options to check if the default options are correctly handled.
    """
    algo = DOEFactory().create("OT_OPT_LHS")
    samples = algo(3, 2)
    assert samples.shape == (3, 2)


@pytest.mark.parametrize(
    "options",
    [
        {"criterion": "unknown_criterion"},
        {"annealing": True, "temperature": "unknown_temperature"},
    ],
)
def test_opt_lhs_wrong_properties(options):
    """Check that using an optimal LHS with wrong properties raises an error."""
    with pytest.raises(ValueError, match="Invalid options for algorithm OT_OPT_LHS"):
        execute_problem(
            doe_algo_name=DOE_LIB_NAME,
            algo_name="OT_OPT_LHS",
            dim=2,
            n_samples=3,
            **options,
        )


def test_centered_lhs():
    """Check that a centered LHS produces samples centered in their cells."""
    algo = DOEFactory().create("OT_LHSC")
    assert set(unique(algo(2, 2)).tolist()) == {0.25, 0.75}


@pytest.mark.parametrize(
    "algo_name,dim,n_samples,options",
    [
        ("OT_MONTE_CARLO", 2, 3, {"n_samples": 3}),
        ("OT_RANDOM", 2, 3, {"n_samples": 3}),
        ("OT_SOBOL", 2, 3, {"n_samples": 3}),
        ("OT_FULLFACT", 3, 8, {"n_samples": 8}),
        (
            "OT_COMPOSITE",
            2,
            33,
            {"levels": [0.1, 0.25, 0.5, 1.0], "centers": [0.2, 0.3]},
        ),
        ("OT_AXIAL", 2, 17, {"levels": [0.1, 0.25, 0.5, 1.0], "centers": [0.2, 0.3]}),
        (
            "OT_FACTORIAL",
            2,
            17,
            {"levels": [0.1, 0.25, 0.5, 1.0], "centers": [0.2, 0.3]},
        ),
        ("OT_LHS", 2, 3, {"n_samples": 3}),
        ("OT_LHSC", 2, 3, {"n_samples": 3}),
        ("OT_OPT_LHS", 2, 3, {"n_samples": 3}),
        ("OT_OPT_LHS", 2, 3, {"n_samples": 3, "temperature": "Linear"}),
        ("OT_OPT_LHS", 2, 3, {"n_samples": 3, "annealing": False}),
        ("OT_OPT_LHS", 2, 3, {"n_samples": 3, "criterion": "PhiP"}),
        ("OT_SOBOL_INDICES", 3, 20, {"n_samples": 20, "eval_second_order": False}),
        ("OT_SOBOL_INDICES", 3, 16, {"n_samples": 20, "eval_second_order": True}),
    ],
)
def test_algos(algo_name, dim, n_samples, options):
    """Check that the OpenTURNS library returns samples correctly shaped."""
    problem = get_problem(dim)
    doe_library = DOEFactory().create(DOE_LIB_NAME)
    doe_library.execute(problem, algo_name=algo_name, dim=dim, **options)
    assert doe_library.unit_samples.shape == (n_samples, dim)


def get_expected_nsamples(
    algo: str,
    dim: int,
    n_samples: int | None = None,
) -> int:
    """Returns the expected number of samples.

    This number depends on the dimension of the problem.

    Args:
       algo: The name of the DOE algorithm.
       dim: The dimension of the variables space.
       n_samples: The number of samples.
           If None, deduce it from the dimension of the variables space.

    Returns:
        The expected number of samples.
    """
    if algo == "OT_AXIAL":
        if dim == 1:
            return 5
        if dim == 5:
            return 21
    if algo == "OT_COMPOSITE":
        if dim == 1:
            return 9
        if dim == 5:
            return 85
    if algo == "OT_FACTORIAL":
        if dim == 1:
            return 5
        if dim == 5:
            return 65
    if algo == "OT_FULLFACT":
        if dim == 5:
            return 1
    if algo == "OT_SOBOL_INDICES":
        if dim == 1:
            return 16
        if dim == 5:
            return 12

    return n_samples


def get_options(
    algo_name: str,
    dim: int,
) -> dict[str, Any]:
    """Returns the options of the algorithms.

    Args:
        algo_name: The name of the DOE algorithm.
        dim: The dimension of the variables spaces.:param algo_name: param dim:

    Returns:
        The options of the DOE algorithm.
    """
    options = {"n_samples": 13}
    if algo_name != "OT_FULLFACT":
        options["levels"] = [0.1, 0.9]
    options["centers"] = [0.5] * dim
    return options


@pytest.mark.parametrize(
    "test_method",
    generate_test_functions(DOE_LIB_NAME, get_expected_nsamples, get_options),
)
def test_methods(test_method):
    """Apply the tests generated by the."""
    test_method()


@pytest.fixture(scope="module")
def variables_space():
    """A variables space."""
    design_space = mock.Mock()
    design_space.variables_names = ["x"]
    design_space.variables_sizes = {"x": 2}
    design_space.dimension = 2
    design_space.untransform_vect = lambda doe, no_check: doe
    return design_space


@pytest.mark.parametrize(
    "name",
    [
        "OT_SOBOL",
        "OT_RANDOM",
        "OT_HASELGROVE",
        "OT_REVERSE_HALTON",
        "OT_HALTON",
        "OT_FAURE",
        "OT_MONTE_CARLO",
        "OT_OPT_LHS",
        "OT_LHS",
        "OT_LHSC",
        "OT_FULLFACT",
        "OT_SOBOL_INDICES",
    ],
)
def test_compute_doe(variables_space, name):
    """Check the computation of a DOE out of a design space."""
    doe = DOEFactory().create(name).compute_doe(variables_space, 4)
    assert doe.shape == (4, variables_space.dimension)


@pytest.mark.parametrize(
    ["name", "size"], [("OT_FACTORIAL", 5), ("OT_COMPOSITE", 9), ("OT_AXIAL", 5)]
)
def test_compute_stratified_doe(variables_space, name, size):
    """Check the computation of a stratified DOE out of a design space."""
    library = DOEFactory().create(name)
    doe = library.compute_doe(
        variables_space, centers=[0.0] * variables_space.dimension, levels=[0.1]
    )
    assert doe.shape == (size, variables_space.dimension)


def test_library_name():
    """Check the library name."""
    assert OpenTURNS.LIBRARY_NAME == "OpenTURNS"


@pytest.mark.parametrize("n_samples", [2, 3, 4])
@pytest.mark.parametrize("seed", [1, 2])
def test_executed_twice(identity_problem, n_samples, seed):
    """Check that the second call to execute() is correctly taken into account."""
    library = OpenTURNS()
    library.execute(
        identity_problem, "OT_MONTE_CARLO", n_samples=3, algo_type="doe", seed=1
    )
    library.execute(
        identity_problem,
        "OT_MONTE_CARLO",
        n_samples=n_samples,
        algo_type="doe",
        seed=seed,
    )
    if seed == 1:
        assert len(identity_problem.database) == max(3, n_samples)
        assert identity_problem.max_iter == n_samples
        assert identity_problem.current_iter == max(n_samples - 3, 0)
    else:
        assert len(identity_problem.database) == 3 + n_samples
        assert identity_problem.max_iter == n_samples
        assert identity_problem.current_iter == n_samples
