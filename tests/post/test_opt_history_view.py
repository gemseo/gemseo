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

from pathlib import Path

import pytest
from numpy import array

from gemseo import execute_algo
from gemseo import execute_post
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.post.opt_history_view import OptHistoryView
from gemseo.utils.testing.helpers import image_comparison

DIR_PATH = Path(__file__).parent
POWER2_PATH = DIR_PATH / "power2_opt_pb.h5"
POWER2_NAN_PATH = DIR_PATH / "power2_opt_pb_nan.h5"


def test_get_constraints() -> None:
    """Test that the constraints of the problem are retrieved correctly."""
    problem = OptimizationProblem.from_hdf(POWER2_PATH)
    view = OptHistoryView(problem)

    assert len(view._OptHistoryView__get_constraint_history(["toto", "ineq1"])) == 1


@pytest.mark.parametrize(
    ("obj_relative", "baseline_images"),
    [
        (
            False,
            [
                "power2_2_variables",
                "power2_2_objective",
                "power2_2_x_xstar",
                "power2_2_ineq_constraints",
                "power2_2_eq_constraints",
            ],
        ),
        (
            True,
            [
                "power2_2_variables",
                "power2_2_objective_relative",
                "power2_2_x_xstar",
                "power2_2_ineq_constraints",
                "power2_2_eq_constraints",
            ],
        ),
    ],
)
@image_comparison(None)
def test_opt_hist_const(baseline_images, obj_relative) -> None:
    """Test that a problem with constraints is properly rendered."""
    problem = OptimizationProblem.from_hdf(POWER2_PATH)
    execute_post(
        problem,
        post_name="OptHistoryView",
        show=False,
        save=False,
        variable_names=["x"],
        file_path="power2_2",
        obj_min=0.0,
        obj_max=5.0,
        obj_relative=obj_relative,
    )


@pytest.mark.parametrize(
    ("problem_path", "baseline_images"),
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
                "power2view_ineq_constraints",
                "power2view_eq_constraints",
            ],
        ),
    ],
)
@image_comparison(None)
def test_opt_hist_from_database(
    baseline_images,
    problem_path,
) -> None:
    """Test the generation of the plots from databases.

    Args:
        baseline_images: The reference images to be compared.
        problem_path: The path to the hdf5 database of the problem to test.
    """
    problem = OptimizationProblem.from_hdf(problem_path)
    # The use of the default value is deliberate;
    # to check that the JSON grammar works properly.
    execute_post(
        problem, post_name="OptHistoryView", variable_names=(), show=False, save=False
    )


TEST_PARAMETERS = {
    "standardized": (
        True,
        [
            "opt_history_view_variables_standardized",
            "opt_history_view_objective_standardized",
            "opt_history_view_x_xstar_standardized",
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
            "opt_history_view_ineq_constraints_unstandardized",
            "opt_history_view_eq_constraints_unstandardized",
        ],
    ),
}


@pytest.mark.parametrize(
    ("use_standardized_objective", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective,
    baseline_images,
    three_length_common_problem,
) -> None:
    """Check OptHistoryView with objective, standardized or not."""
    opt = OptHistoryView(three_length_common_problem)
    three_length_common_problem.use_standardized_objective = use_standardized_objective
    opt.execute(save=False)


@pytest.mark.parametrize(
    ("case", "baseline_images"),
    [
        (
            1,
            [
                "461_1_opt_history_view_variables",
                "461_1_opt_history_view_objective",
                "461_1_opt_history_view_x_xstar",
            ],
        ),
        (
            2,
            [
                "461_2_opt_history_view_variables",
                "461_2_opt_history_view_objective",
                "461_2_opt_history_view_x_xstar",
            ],
        ),
    ],
)
@image_comparison(None)
def test_461(case, baseline_images) -> None:
    """Check that OptHistoryView works with the cases mentioned in issue 461.

    1. Design space of dimension 1 and scalar output.
    2. Design space of dimension > 1 and vector output.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-2, upper_bound=2.0, value=-2.0)
    if case == 2:
        design_space.add_variable("y", lower_bound=-2, upper_bound=2.0, value=-2.0)

    problem = OptimizationProblem(design_space)
    if case == 1:
        problem.objective = MDOFunction(lambda x: x[0] ** 2, "func")
    elif case == 2:
        problem.objective = problem.objective = MDOFunction(
            lambda x: array([x[0] ** 2 + x[1] ** 2]), "func"
        )
    problem.differentiation_method = problem.ApproximationMode.FINITE_DIFFERENCES

    execute_algo(problem, algo_name="NLOPT_SLSQP", max_iter=5)
    execute_post(problem, post_name="OptHistoryView", save=False, show=False)


@image_comparison(
    baseline_images=[
        "opt_history_view_variables_variable_names",
        "opt_history_view_objective_variable_names",
        "opt_history_view_x_xstar_variable_names",
        "opt_history_view_ineq_constraints_variable_names",
    ]
)
def test_variable_names() -> None:
    execute_post(
        Path(__file__).parent / "mdf_backup.h5",
        post_name="OptHistoryView",
        variable_names=["x_2", "x_1"],
        save=False,
        show=False,
    )


def test_no_gradient_history(caplog) -> None:
    """Check that OptHistoryView works without gradient history."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x**2, "f")
    problem.database.store(array([-1]), {"f": array([1])})
    problem.database.store(array([0]), {"f": array([0])})
    problem.database.store(array([1]), {"f": array([1])})
    post_processor = execute_post(
        problem, post_name="OptHistoryView", save=False, show=False
    )
    assert set(post_processor.figures.keys()) == {"variables", "objective", "x_xstar"}
