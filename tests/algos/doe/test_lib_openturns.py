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

import logging
import re
from typing import Any
from unittest import mock

import pytest
from numpy import array
from numpy import unique
from numpy.testing import assert_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe._openturns.ot_axial_doe import OTAxialDOE
from gemseo.algos.doe._openturns.ot_composite_doe import OTCompositeDOE
from gemseo.algos.doe._openturns.ot_factorial_doe import OTFactorialDOE
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.doe.lib_openturns import OpenTURNS
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.mdofunctions.mdo_function import MDOFunction

from .utils import execute_problem
from .utils import generate_test_functions
from .utils import get_problem

DOE_LIB_NAME = "OpenTURNS"


@pytest.fixture()
def identity_problem() -> OptimizationProblem:
    """A problem whose objective is the identity function defined over [0,1]."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "f")
    return problem


def test_library_from_factory() -> None:
    """Check that the DOEFactory can create the OpenTURNS library."""
    factory = DOEFactory()
    if factory.is_available(DOE_LIB_NAME):
        factory.create(DOE_LIB_NAME)


def test_call() -> None:
    """Check that calling OpenTURNS library returns a NumPy array correctly shaped.

    Use an algorithm with options to check if the default options are correctly handled.
    """
    algo = DOEFactory().create("OT_OPT_LHS")
    samples = algo(3, 2)
    assert samples.shape == (3, 2)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        (
            {"criterion": "unknown_criterion"},
            "data.criterion must be one of ['C2', 'PhiP', 'MinDist']",
        ),
        (
            {"annealing": True, "temperature": "unknown_temperature"},
            "data.temperature must be one of ['Geometric', 'Linear']",
        ),
    ],
)
def test_opt_lhs_wrong_properties(options, error) -> None:
    """Check that using an optimal LHS with wrong properties raises an error."""
    match = re.escape(
        "Grammar OT_OPT_LHS_algorithm_options: validation failed." f"\nerror: {error}"
    )
    with pytest.raises(InvalidDataError, match=match):
        execute_problem(
            doe_algo_name=DOE_LIB_NAME,
            algo_name="OT_OPT_LHS",
            dim=2,
            n_samples=3,
            **options,
        )


def test_centered_lhs() -> None:
    """Check that a centered LHS produces samples centered in their cells."""
    algo = DOEFactory().create("OT_LHSC")
    assert set(unique(algo(2, 2)).tolist()) == {0.25, 0.75}


@pytest.mark.parametrize(
    ("algo_name", "dim", "n_samples", "options"),
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
def test_algos(algo_name, dim, n_samples, options) -> None:
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
    if algo == "OT_FULLFACT" and dim == 5:
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
    options = {}
    if algo_name in ["OT_AXIAL", "OT_COMPOSITE", "OT_FACTORIAL"]:
        options["levels"] = [0.1, 0.9]
    else:
        options["n_samples"] = 13

    if algo_name == "OT_FULLFACT":
        options["centers"] = [0.5] * dim

    return options


@pytest.mark.parametrize(
    "test_method",
    generate_test_functions(DOE_LIB_NAME, get_expected_nsamples, get_options),
)
def test_methods(test_method) -> None:
    """Apply the tests generated by the."""
    test_method()


@pytest.fixture(scope="module")
def variables_space():
    """A variables space."""
    design_space = mock.Mock()
    design_space.variable_names = ["x"]
    design_space.variable_sizes = {"x": 2}
    design_space.dimension = 2
    design_space.untransform_vect = lambda doe, no_check: doe
    design_space.normalize = {"x": array([True, True])}
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
def test_compute_doe(variables_space, name) -> None:
    """Check the computation of a DOE out of a design space."""
    doe = DOEFactory().create(name).compute_doe(variables_space, 4)
    assert doe.shape == (4, variables_space.dimension)


@pytest.mark.parametrize(
    ("name", "size"), [("OT_FACTORIAL", 5), ("OT_COMPOSITE", 9), ("OT_AXIAL", 5)]
)
def test_compute_stratified_doe(variables_space, name, size) -> None:
    """Check the computation of a stratified DOE out of a design space."""
    library = DOEFactory().create(name)
    doe = library.compute_doe(
        variables_space, centers=[0.5] * variables_space.dimension, levels=[0.1]
    )
    assert doe.shape == (size, variables_space.dimension)


def test_library_name() -> None:
    """Check the library name."""
    assert OpenTURNS.LIBRARY_NAME == "OpenTURNS"


@pytest.mark.parametrize("n_samples", [2, 3, 4])
@pytest.mark.parametrize("seed", [1, 2])
def test_executed_twice(identity_problem, n_samples, seed) -> None:
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


def test_optimized_lhs_size_1():
    """Check that size 1 is not allowed for optimized LHS."""
    library = OpenTURNS()
    library.init_options_grammar("OT_OPT_LHS")
    with pytest.raises(
        InvalidDataError, match="data.n_samples must be bigger than or equal to 2"
    ):
        library._get_options(n_samples=1)


@pytest.mark.parametrize(
    ("cls", "error_message"),
    [
        (
            OTAxialDOE,
            "An axial DOE in dimension d=2 requires at least 1+2*d=5 samples; got 4.",
        ),
        (
            OTCompositeDOE,
            (
                "A composite DOE in dimension d=2 requires "
                "at least 1+2*d+2^d=9 samples; "
                "got 4."
            ),
        ),
        (
            OTFactorialDOE,
            (
                "A factorial DOE in dimension d=2 requires at least 1+2^d=5 samples; "
                "got 4."
            ),
        ),
    ],
)
def test_ot_stratified_doe_n_samples_error(cls, error_message):
    """Verify that a stratified DOE algorithm raises when n_samples is too small."""
    with pytest.raises(ValueError, match=re.escape(error_message)):
        cls().generate_samples(4, 2)


@pytest.mark.parametrize(
    ("cls", "n_samples"), [(OTAxialDOE, 25), (OTCompositeDOE, 57), (OTFactorialDOE, 33)]
)
def test_ot_stratified_doe_n_samples(cls, n_samples):
    """Verify that a stratified DOE algorithm works when n_samples is ok."""
    # We use d=3 and l=4.
    assert cls().generate_samples(n_samples, 3).shape == (n_samples, 3)


@pytest.mark.parametrize(
    ("cls", "n_samples", "error_message"),
    [
        (
            OTAxialDOE,
            26,
            (
                "An axial DOE of 26 samples in dimension 3 does not exist; "
                "use 25 samples instead."
            ),
        ),
        (
            OTCompositeDOE,
            58,
            (
                "A composite DOE of 58 samples in dimension 3 does not exist; "
                "use 57 samples instead."
            ),
        ),
        (
            OTFactorialDOE,
            34,
            (
                "A factorial DOE of 34 samples in dimension 3 does not exist; "
                "use 33 samples instead."
            ),
        ),
    ],
)
def test_ot_stratified_doe_n_samples_warning(cls, n_samples, caplog, error_message):
    """Verify that a stratified DOE algorithm warns when n_samples is too important."""
    # We use d=3 and l=4.
    algo = cls()
    samples = algo.generate_samples(n_samples, 3)
    assert samples.shape == (n_samples - 1, 3)

    _, level, msg = caplog.record_tuples[0]
    assert level == logging.WARNING
    assert msg == error_message

    assert_equal(samples, cls().generate_samples(n_samples - 1, 3))


@pytest.mark.parametrize("cls", [OTAxialDOE, OTCompositeDOE, OTFactorialDOE])
def test_ot_stratified_centers_dimension_error(cls):
    """Verify that a stratified DOE algorithm raises when centers is ill-dimensioned."""
    with pytest.raises(
        ValueError, match=re.escape("The number of centers must be 3; got 2.")
    ):
        cls().generate_samples(None, 3, centers=[0.5, 0.6])


@pytest.mark.parametrize("cls", [OTAxialDOE, OTCompositeDOE, OTFactorialDOE])
@pytest.mark.parametrize("centers", [[0, 0.5], [0.5, 1]])
def test_ot_stratified_centers_value_error(cls, centers):
    """Verify that a stratified DOE algorithm raises when centers has wrong values."""
    with pytest.raises(
        ValueError, match=re.escape(f"The centers must be in ]0,1[; got {centers}.")
    ):
        cls().generate_samples(None, 2, centers=centers)


@pytest.mark.parametrize("cls", [OTAxialDOE, OTCompositeDOE, OTFactorialDOE])
def test_ot_stratified_levels_value_error(cls):
    """Verify that a stratified DOE algorithm raises when centers has wrong values."""
    with pytest.raises(
        ValueError, match=re.escape("The levels must be in ]0,1]; got [0, 0.5].")
    ):
        cls().generate_samples(None, 2, levels=[0, 0.5])


@pytest.mark.parametrize(
    ("cls", "centers", "levels", "expected_samples"),
    [
        (
            OTAxialDOE,
            0.5,
            [0.5],
            array([[0.5, 0.5], [0.75, 0.5], [0.25, 0.5], [0.5, 0.75], [0.5, 0.25]]),
        ),
        (
            OTAxialDOE,
            [0.5],
            [0.5],
            array([[0.5, 0.5], [0.75, 0.5], [0.25, 0.5], [0.5, 0.75], [0.5, 0.25]]),
        ),
        (
            OTAxialDOE,
            [0.75, 0.5],
            [0.5],
            array([
                [0.75, 0.5],
                [0.875, 0.5],
                [0.375, 0.5],
                [0.75, 0.75],
                [0.75, 0.25],
            ]),
        ),
        (
            OTCompositeDOE,
            0.5,
            [0.5],
            array([
                [0.5, 0.5],
                [0.25, 0.25],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
                [0.75, 0.5],
                [0.25, 0.5],
                [0.5, 0.75],
                [0.5, 0.25],
            ]),
        ),
        (
            OTCompositeDOE,
            [0.5],
            [0.5],
            array([
                [0.5, 0.5],
                [0.25, 0.25],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.75, 0.75],
                [0.75, 0.5],
                [0.25, 0.5],
                [0.5, 0.75],
                [0.5, 0.25],
            ]),
        ),
        (
            OTCompositeDOE,
            [0.75, 0.5],
            [0.5],
            array([
                [0.75, 0.5],
                [0.375, 0.25],
                [0.875, 0.25],
                [0.375, 0.75],
                [0.875, 0.75],
                [0.875, 0.5],
                [0.375, 0.5],
                [0.75, 0.75],
                [0.75, 0.25],
            ]),
        ),
        (
            OTFactorialDOE,
            0.5,
            [0.5],
            array([[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]),
        ),
        (
            OTFactorialDOE,
            [0.5],
            [0.5],
            array([[0.5, 0.5], [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]),
        ),
        (
            OTFactorialDOE,
            [0.7, 0.5],
            [0.5],
            array([[0.7, 0.5], [0.35, 0.25], [0.85, 0.25], [0.35, 0.75], [0.85, 0.75]]),
        ),
    ],
)
def test_ot_stratified_results(cls, centers, levels, expected_samples):
    """Check the DOEs generated by the stratified DOE algorithms."""
    samples = cls().generate_samples(None, 2, centers=centers, levels=levels)
    assert_equal(samples, expected_samples)
