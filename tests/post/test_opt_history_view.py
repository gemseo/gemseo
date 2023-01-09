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
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import execute_algo
from gemseo.api import execute_post
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.post.opt_history_view import OptHistoryView
from gemseo.utils.testing import image_comparison
from numpy import array

DIR_PATH = Path(__file__).parent
POWER2_PATH = DIR_PATH / "power2_opt_pb.h5"
POWER2_NAN_PATH = DIR_PATH / "power2_opt_pb_nan.h5"


def test_get_constraints():
    """Test that the constraints of the problem are retrieved correctly."""
    problem = OptimizationProblem.import_hdf(POWER2_PATH)
    view = OptHistoryView(problem)

    _, cstr = view._get_constraints(["toto", "ineq1"])
    assert len(cstr) == 1


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python 3.8 or greater")
@image_comparison(
    [
        "power2_2_variables",
        "power2_2_objective",
        "power2_2_x_xstar",
        "power2_2_hessian_approximation",
        "power2_2_ineq_constraints",
        "power2_2_eq_constraints",
    ]
)
def test_opt_hist_const(pyplot_close_all):
    """Test that a problem with constraints is properly rendered.

    Args:
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = OptimizationProblem.import_hdf(POWER2_PATH)
    post = execute_post(
        problem,
        "OptHistoryView",
        show=False,
        save=False,
        variables_names=["x"],
        file_path="power2_2",
        obj_min=0.0,
        obj_max=5.0,
    )
    post.figures


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python 3.8 or greater")
@pytest.mark.parametrize(
    "problem_path,baseline_images",
    [
        (
            POWER2_NAN_PATH,
            [
                "opt_history_view_variables_nan",
                "opt_history_view_objective_nan",
                "opt_history_view_x_xstar_nan",
                "opt_history_view_ineq_constraints_nan",
                "opt_history_view_eq_constraints_nan",
            ],
        ),
        (
            POWER2_PATH,
            [
                "power2view_variables",
                "power2view_objective",
                "power2view_x_xstar",
                "power2view_hessian_approximation",
                "power2view_ineq_constraints",
                "power2view_eq_constraints",
            ],
        ),
    ],
)
@image_comparison(None)
def test_opt_hist_from_database(baseline_images, problem_path, pyplot_close_all):
    """Test the generation of the plots from databases.

    Args:
        baseline_images: The reference images to be compared.
        problem_path: The path to the hdf5 database of the problem to test.
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = OptimizationProblem.import_hdf(problem_path)
    post = execute_post(problem, "OptHistoryView", show=False, save=False)
    post.figures


def test_diag_with_nan(caplog):
    """Check that the Hessian plot creation is skipped if its diagonal contains NaN."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(
        lambda x: 2 * x, "obj", jac=lambda x: array([[2.0]])
    )
    execute_algo(problem, "fullfact", n_samples=3, eval_jac=True, algo_type="doe")

    execute_post(problem, "OptHistoryView", save=False, show=False)
    log = caplog.text
    assert "Failed to create Hessian approximation." in log
    assert "ValueError: The approximated Hessian diagonal contains NaN." in log


TEST_PARAMETERS = {
    "standardized": (
        True,
        [
            "opt_history_view_variables_standardized",
            "opt_history_view_objective_standardized",
            "opt_history_view_x_xstar_standardized",
            "opt_history_view_hessian_approximation_standardized",
            "opt_history_view_ineq_constraints_standardized",
            "opt_history_view_eq_constraints_standardized",
        ],
    ),
    "unstandardized": (
        False,
        [
            "opt_history_view_variables_unstandardized",
            "opt_history_view_objective_unstandardized",
            "opt_history_view_x_xstar_unstandardized",
            "opt_history_view_hessian_approximation_unstandardized",
            "opt_history_view_ineq_constraints_unstandardized",
            "opt_history_view_eq_constraints_unstandardized",
        ],
    ),
}


@pytest.mark.parametrize(
    "use_standardized_objective, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective,
    baseline_images,
    three_length_common_problem,
    pyplot_close_all,
):
    """Check OptHistoryView with objective, standardized or not."""
    opt = OptHistoryView(three_length_common_problem)
    three_length_common_problem.use_standardized_objective = use_standardized_objective
    opt.execute(save=False)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python 3.8 or greater")
@pytest.mark.parametrize(
    "case,baseline_images",
    [
        (
            1,
            [
                "461_1_opt_history_view_variables",
                "461_1_opt_history_view_objective",
                "461_1_opt_history_view_x_xstar",
                "461_1_opt_history_view_hessian",
            ],
        ),
        (
            2,
            [
                "461_2_opt_history_view_variables",
                "461_2_opt_history_view_objective",
                "461_2_opt_history_view_x_xstar",
                "461_2_opt_history_view_hessian",
            ],
        ),
    ],
)
@image_comparison(None)
def test_461(case, baseline_images):
    """Check that OptHistoryView works with the cases mentioned in issue 461.

    1. Design space of dimension 1 and scalar output.
    2. Design space of dimension > 1 and vector output.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=-2, u_b=2.0, value=-2.0)
    if case == 2:
        design_space.add_variable("y", l_b=-2, u_b=2.0, value=-2.0)

    problem = OptimizationProblem(design_space)
    if case == 1:
        problem.objective = MDOFunction(lambda x: x[0] ** 2, "func")
    elif case == 2:
        problem.objective = problem.objective = MDOFunction(
            lambda x: array([x[0] ** 2 + x[1] ** 2]), "func"
        )
    problem.differentiation_method = problem.FINITE_DIFFERENCES

    execute_algo(problem, "NLOPT_SLSQP", max_iter=5)
    execute_post(problem, "OptHistoryView", save=False, show=False)
