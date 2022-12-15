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
from __future__ import annotations

from pathlib import Path

import pytest
from gemseo.algos.database import Database
from gemseo.api import execute_algo

p7core = pytest.importorskip("da.p7core", reason="pSeven is not available")

from numpy import array, ones  # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402

from gemseo.algos.design_space import DesignSpace  # noqa: E402
from gemseo.algos.opt.opt_factory import OptimizersFactory  # noqa: E402
from gemseo.algos.opt_problem import OptimizationProblem  # noqa: E402
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction  # noqa: E402
from gemseo.problems.analytical.power_2 import Power2  # noqa: E402
from gemseo.problems.analytical.rosenbrock import Rosenbrock  # noqa: E402


def check_on_problem(
    problem: Rosenbrock | Power2,
    algo_name: str,
    **options,
) -> OptimizationProblem:
    """Check that a pSeven optimizer solves a given problem.

    Args:
        problem: The optimization problem.
        algo_name: The name of the pSeven algorithm.
        options: The options of the algorithm.

    Returns:
        The solved optimization problem.
    """
    x_opt, f_opt = problem.get_solution()
    result = OptimizersFactory().execute(problem, algo_name, **options)
    assert result.f_opt == pytest.approx(f_opt, abs=5e-3)
    assert_allclose(result.x_opt, x_opt, rtol=0.1)
    return problem


@pytest.mark.parametrize(
    "algo_name,algo_options",
    [
        ("PSEVEN", {"max_iter": 40, "normalize_design_space": False}),
        ("PSEVEN", {"max_iter": 40, "normalize_design_space": True}),
        ("PSEVEN", {"max_iter": 100, "evaluation_cost_type": "Expensive"}),
        ("PSEVEN", {"max_iter": 40, "batch_size": 1}),
        ("PSEVEN", {"max_iter": 40, "use_threading": False}),
        ("PSEVEN", {"max_iter": 40, "use_threading": True}),
        ("PSEVEN_FD", {"max_iter": 40}),
        ("PSEVEN_NCG", {"max_iter": 70}),
        ("PSEVEN_NLS", {"max_iter": 80}),
        ("PSEVEN_POWELL", {"max_iter": 160}),
    ],
)
def test_pseven_rosenbrock(algo_name, algo_options):
    """Check that pSeven's optimizers minimize Rosenbrock's function."""
    check_on_problem(Rosenbrock(), algo_name, **algo_options)


@pytest.mark.parametrize("normalize_design_space", [False, True])
def test_pseven_power2(normalize_design_space):
    """Check that pSeven's default optimizer solves the Power2 problem."""
    check_on_problem(
        Power2(),
        "PSEVEN",
        max_iter=10,
        normalize_design_space=normalize_design_space,
        eq_tolerance=1e-4,
    )


@pytest.mark.parametrize(
    "algo_name", ["PSEVEN_MOM", "PSEVEN_QP", "PSEVEN_SQP", "PSEVEN_SQ2P"]
)
def test_pseven_unconstrained(algo_name):
    """Check the optimiers that cannot be run on an unconstrained problem."""
    with pytest.raises(
        RuntimeError, match=f"{algo_name} requires at least one constraint"
    ):
        OptimizersFactory().execute(Rosenbrock(), algo_name)


@pytest.mark.parametrize(
    ["evaluation_cost_type"], [({"rosen": "Affordable"},), ("Affordable",)]
)
def test_evaluation_cost_type_invalid(evaluation_cost_type):
    """Check the passing of invalid evaluation cost types."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(
            Rosenbrock(), "PSEVEN", evaluation_cost_type=evaluation_cost_type
        )


def test_expensive_evaluations():
    """Check the passing of numbers of expensive evaluations."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(
            Rosenbrock(), "PSEVEN", expensive_evaluations={"rosen": 0.1}
        )


def test_objectives_smoothness():
    """Check the passing of an objectives smoothness hint."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(
            Rosenbrock(), "PSEVEN", objectives_smoothness="tata"
        )


def test_constraints_smoothness():
    """Check the passing of a constraints smoothness hint."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(
            Rosenbrock(), "PSEVEN", constraints_smoothness="toto"
        )


def test_log_level():
    """Check the passing of a log level."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", log_level="High")


def test_diff_scheme():
    """Check the passing of a differentiation scheme order."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", diff_scheme="ThirdOrder")


def test_diff_type():
    """Check the passing of a differentiation scheme type."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", diff_type="Complex")


def test_globalization_method():
    """Check the passing of a globalization method."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
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


def test_pseven_techniques():
    """Check the passing of pSeven techniques."""
    lib = OptimizersFactory().create("PSEVEN_FD")
    lib.init_options_grammar("PSEVEN_FD")
    lib.problem = Rosenbrock()
    options = lib._get_options(globalization_method="RL", surrogate_based=True)
    assert options["GTOpt/Techniques"] == "[FD, RL, SBO]"


@pytest.mark.parametrize(
    ["options", "message"],
    [
        ({"max_iter": 2}, "Maximum number of iterations reached."),
        (
            {"xtol_abs": 1e6},
            "Successive iterates of the design variables are closer than xtol_rel"
            " or xtol_abs.",
        ),
        (
            {"ftol_abs": 1e6},
            "Successive iterates of the objective function are closer than ftol_rel"
            " or ftol_abs.",
        ),
    ],
)
def test_gemseo_stopping(options, message):
    """Check the termination of the optimization by GEMSEO."""
    result = OptimizersFactory().execute(Rosenbrock(), "PSEVEN", **options)
    assert result.message == message + " GEMSEO Stopped the driver"


def test_pseven_stop_before_gemseo():
    """Check the termination of the optimization by pSeven rather than Gemseo."""
    result = OptimizersFactory().execute(
        Rosenbrock(),
        "PSEVEN",
        evaluation_cost_type={"rosen": "Expensive"},
        expensive_evaluations={"rosen": 2},
    )
    assert result.status == 0
    assert result.message == "Success"


def test_pseven_sample_x():
    """Check that the input sample is evaluated."""
    current_x = Rosenbrock().design_space.get_current_value()
    sample_x = [array([2.0, -2.0]), array([-2.0, 2.0])]
    problem = Rosenbrock()
    execute_algo(problem, "PSEVEN", sample_x=sample_x, max_iter=3)
    database = problem.database
    assert database.contains_x(current_x)
    assert database.contains_x(sample_x[0])
    assert database.contains_x(sample_x[1])


@pytest.mark.parametrize("global_phase_intensity", ["toto", -1, 1.1])
def test_global_phase_intensity(global_phase_intensity):
    """Check the "global phase intensity" option."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(
            Rosenbrock(), "PSEVEN", global_phase_intensity=global_phase_intensity
        )


@pytest.mark.parametrize("deterministic", ["Yes", 0])
def test_deterministic(deterministic):
    """Check the "deterministic" option."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", deterministic=deterministic)


def test_local_search():
    """Check the "local search" option."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", local_search="Yes")


def test_responses_scalability():
    """Check the "responses scalability" option."""
    with pytest.raises(ValueError, match="Invalid options for algorithm PSEVEN"):
        OptimizersFactory().execute(Rosenbrock(), "PSEVEN", responses_scalability=0)


@pytest.mark.parametrize("use_gradient", [False, True])
def test_disable_derivatives(use_gradient):
    """Check the disabling of the derivatives."""
    problem = check_on_problem(Rosenbrock(), "PSEVEN", use_gradient=use_gradient)
    gradient_name = f"{Database.GRAD_TAG}{problem.objective.name}"
    assert (
        any(gradient_name in values for values in problem.database.values())
        == use_gradient
    )


def test_log_file(tmp_wd):
    """Check the log file."""
    path = Path("log.txt")
    assert not path.is_file()
    OptimizersFactory().execute(
        Rosenbrock(), "PSEVEN", log_level="Info", log_path=str(path)
    )
    assert path.is_file()
    # Check that the file is not empty
    assert path.stat().st_size > 0


def test_expensive_iterations_warning(caplog):
    """Check the warning on too small evaluations budget for expensive evaluations."""
    OptimizersFactory().execute(
        Rosenbrock(),
        "PSEVEN",
        evaluation_cost_type="Expensive",
        max_iter=1,
        max_expensive_func_iter=1,
    )
    message = (
        "The evaluations budget (max_iter=1) is to small to compute the "
        "expensive functions at both the initial guesses (1) and the iterates "
        "(1)."
    )
    assert message in caplog.text


def test_library_name():
    """Check the library name."""
    from gemseo.algos.opt.lib_pseven import PSevenOpt

    assert PSevenOpt.LIBRARY_NAME == "pSeven"
