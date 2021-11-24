# -*- coding: utf-8 -*-
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
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""Tests for the Generic Tool for Optimization (GTOpt) of pSeven Core."""

from __future__ import unicode_literals

from typing import Union
from unittest import mock

import pytest

from gemseo.api import execute_algo

p7core = pytest.importorskip("da.p7core", reason="pSeven is not available")

from numpy import array, ones  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from gemseo.algos.design_space import DesignSpace  # noqa: E402
from gemseo.algos.opt.core.pseven_problem_adapter import CostType  # noqa: E402
from gemseo.algos.opt.lib_pseven import GlobalMethod  # noqa: E402
from gemseo.algos.opt.opt_factory import OptimizersFactory  # noqa: E402
from gemseo.algos.opt_problem import OptimizationProblem  # noqa: E402
from gemseo.core.mdofunctions.mdo_function import MDOLinearFunction  # noqa: E402
from gemseo.problems.analytical.power_2 import Power2  # noqa: E402
from gemseo.problems.analytical.rosenbrock import Rosenbrock  # noqa: E402


def check_on_problem(
    problem,  # type: Union[Rosenbrock, Power2]
    algo_name,  # type: str
    **options
):  # type: (...) -> None
    """Check that a pSeven optimizer solves a given problem.

    Args:
        problem: The optimization problem.
        algo_name: The name of the pSeven algorithm.
        options: The options of the algorithm.
    """
    x_opt, f_opt = problem.get_solution()
    result = OptimizersFactory().execute(problem, algo_name, **options)
    assert result.f_opt == pytest.approx(f_opt, abs=1e-6)
    assert_allclose(result.x_opt, x_opt, rtol=1e-3)


@pytest.mark.parametrize(
    "algo_name,algo_options",
    [
        ("PSEVEN", {}),
        ("PSEVEN_FD", {}),
        ("PSEVEN_NCG", {}),
        ("PSEVEN_NLS", {}),
        ("PSEVEN_POWELL", {"max_iter": 200}),
    ],
)
def test_pseven_rosenbrock(algo_name, algo_options):
    """Check that pSeven's optimizers minimize Rosenbrock's function."""
    check_on_problem(Rosenbrock(), algo_name, **algo_options)


def test_pseven_power2():
    """Check that pSeven's default optimizer solves the Power2 problem."""
    check_on_problem(Power2(), "PSEVEN")


@pytest.mark.parametrize(
    "algo_name", ["PSEVEN_MOM", "PSEVEN_QP", "PSEVEN_SQP", "PSEVEN_SQ2P"]
)
def test_pseven_unconstrained(algo_name):
    """Check the optimiers that cannot be run on an unconstrained problem."""
    with pytest.raises(
        RuntimeError, match="{} requires at least one constraint".format(algo_name)
    ):
        OptimizersFactory().execute(Rosenbrock(), algo_name)


@pytest.mark.parametrize(
    ["name", "cost_type", "message"],
    [
        ("f", CostType.EXPENSIVE, "Unknown function name: f"),
        ("rosen", "Affordable", "Unknown cost type for function 'rosen': Affordable"),
    ],
)
def test_evaluation_cost_type(name, cost_type, message):
    """Check the passing of evaluation cost types."""
    with pytest.raises(ValueError, match=message):
        OptimizersFactory().execute(
            Rosenbrock(), "PSEVEN", evaluation_cost_type={name: cost_type}
        )


@pytest.mark.parametrize(
    ["name", "number", "error", "message"],
    [
        ("f", 10, ValueError, "Unknown function name: f"),
        (
            "rosen",
            0.1,
            TypeError,
            "Non-integer evaluations number for function 'rosen': 0.1",
        ),
    ],
)
def test_expensive_evaluations(name, number, error, message):
    """Check the passing of numbers of expensive evaluations."""
    with pytest.raises(error, match=message):
        OptimizersFactory().execute(
            Rosenbrock(), "PSEVEN", expensive_evaluations={name: number}
        )


def test_objectives_smoothness():
    """Check the passing of an objectives smoothness hint."""
    with pytest.raises(ValueError, match="Unknown objectives smoothness: tata"):
        OptimizersFactory().execute(
            Rosenbrock(), "PSEVEN", objectives_smoothness="tata"
        )


def test_constraints_smoothness():
    """Check the passing of a constraints smoothness hint."""
    with pytest.raises(ValueError, match="Unknown constraints smoothness: toto"):
        OptimizersFactory().execute(
            Rosenbrock(), "PSEVEN", constraints_smoothness="toto"
        )


def test_log_level():
    """Check the passing of a log level."""
    with pytest.raises(ValueError, match="Unknown log level: High"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", log_level="High")


def test_diff_scheme():
    """Check the passing of a differentiation scheme order."""
    with pytest.raises(ValueError, match="Unknown differentiation scheme: ThirdOrder"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", diff_scheme="ThirdOrder")


def test_diff_type():
    """Check the passing of a differentiation scheme type."""
    with pytest.raises(ValueError, match="Unknown differentiation type: Complex"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", diff_type="Complex")


def test_globalization_method():
    """Check the passing of a globalization method."""
    with pytest.raises(ValueError, match="Unknown globalization method: TR"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", globalization_method="TR")


def test_qp_nonquadratic_objective():
    """Check the call of the QP algorithm on a non-quadratic objective."""
    problem = Rosenbrock()
    problem.add_ineq_constraint(MDOLinearFunction(ones(2), "g", value_at_zero=1.0))
    with pytest.raises(
        TypeError, match="PSEVEN_QP requires the objective to be quadratic or linear"
    ):
        OptimizersFactory().execute(problem, "PSEVEN_QP")


def test_qp_nonlinear_constraint():
    """Check the call of the QP algorithm oon a nonlinear constraint."""
    design_space = DesignSpace()
    design_space.add_variable("x", 2, value=0.0)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOLinearFunction(-ones(2), "f")
    problem.add_ineq_constraint(Rosenbrock().objective)

    with pytest.raises(
        TypeError,
        match="PSEVEN_QP requires the constraints to be linear,"
        " the following is not: rosen",
    ):
        OptimizersFactory().execute(problem, "PSEVEN_QP")


@pytest.mark.parametrize(
    ["option_value", "pseven_value"], [(True, "Forced"), (False, "Disabled")]
)
def test_local_search_option(option_value, pseven_value):
    """Check the passing of the local search option."""
    lib = OptimizersFactory().create("PSEVEN")
    lib.init_options_grammar("PSEVEN")
    lib.problem = Rosenbrock()
    options = lib._get_options(local_search=option_value)
    assert options["GTOpt/LocalSearch"] == pseven_value


@pytest.mark.parametrize(
    ["has_eq", "has_ineq", "tolerance"],
    [
        (True, True, 5e-7),
        (True, False, 5e-4),
        (False, True, 5e-7),
        (False, False, None),
    ],
)
def test_constraints_tolerance(has_eq, has_ineq, tolerance):
    """Check the setting of the pSeven tolerance on the constraints."""
    lib = OptimizersFactory().create("PSEVEN")
    lib.init_options_grammar("PSEVEN")
    design_space = DesignSpace()
    design_space.add_variable("x", 3, value=0)
    problem = OptimizationProblem(design_space)
    problem.constraints = [lambda x: x, lambda x: x[0] ** 2]
    problem.eq_tolerance = 1e-3
    problem.ineq_tolerance = 1e-6
    problem.has_eq_constraints = mock.MagicMock(return_value=has_eq)
    problem.has_ineq_constraints = mock.MagicMock(return_value=has_ineq)
    lib.problem = problem
    assert lib._get_options()["GTOpt/ConstraintsTolerance"] == tolerance


def test_pseven_techniques():
    """Check the passing of pSeven techniques."""
    lib = OptimizersFactory().create("PSEVEN_FD")
    lib.init_options_grammar("PSEVEN_FD")
    lib.problem = Rosenbrock()
    options = lib._get_options(
        globalization_method=GlobalMethod.RL, surrogate_based=True
    )
    assert options["GTOpt/Techniques"] == "[FD, RL, SBO]"


def test_gemseo_stops_before_pseven():
    """Check the termination of the optimization by Gemseo rather than pSeven."""
    result = OptimizersFactory().execute(Rosenbrock(), "PSEVEN", max_iter=1)
    assert result.status == 7
    assert result.message == "User terminated"


def test_pseven_stop_before_gemseo():
    """Check the termination of the optimization by pSeven rather than Gemseo."""
    result = OptimizersFactory().execute(
        Rosenbrock(),
        "PSEVEN",
        evaluation_cost_type={"rosen": CostType.EXPENSIVE},
        expensive_evaluations={"rosen": 2},
    )
    assert result.status == 0
    assert result.message == "Success"


def test_pseven_sample_x():
    """Check that the input sample is evaluated."""
    current_x = Rosenbrock().design_space.get_current_x()
    sample_x = [array([2.0, -2.0]), array([-2.0, 2.0])]
    problem = Rosenbrock()
    execute_algo(problem, "PSEVEN", sample_x=sample_x, max_iter=3)
    database = problem.database
    assert database.contains_x(current_x)
    assert database.contains_x(sample_x[0])
    assert database.contains_x(sample_x[1])
