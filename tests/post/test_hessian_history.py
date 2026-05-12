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
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.post import HessianHistory_Settings
from gemseo.post.hessian_history import HessianHistory
from gemseo.utils.testing.helpers import assert_exception

DIR_PATH = Path(__file__).parent
POWER2_PATH = DIR_PATH / "power2_opt_pb.h5"
POWER2_NAN_PATH = DIR_PATH / "power2_opt_pb_nan.h5"


@pytest.mark.parametrize(
    "obj_relative",
    [False, True],
)
def test_opt_hist_const(obj_relative, snapshot_matplotlib) -> None:
    """Test that a problem with constraints is properly rendered."""
    problem = OptimizationProblem.from_hdf(POWER2_PATH)
    execute_post(
        problem,
        post_name="HessianHistory",
        show=False,
        save=False,
        variable_names=["x"],
    )


@pytest.mark.parametrize(
    "problem_path",
    [POWER2_PATH],
)
def test_opt_hist_from_database(
    problem_path,
    snapshot_matplotlib,
) -> None:
    """Test the generation of the plots from databases.

    Args:
        problem_path: The path to the hdf5 database of the problem to test.
    """
    problem = OptimizationProblem.from_hdf(problem_path)
    # The use of the default value is deliberate;
    # to check that the JSON grammar works properly.
    execute_post(
        problem, post_name="HessianHistory", variable_names=(), show=False, save=False
    )


def test_diag_with_nan(snapshot) -> None:
    """Check that the Hessian plot creation is skipped if its diagonal contains NaN."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(
        lambda x: 2 * x, name="obj", jac=lambda x: array([[2.0]])
    )
    execute_algo(
        problem, algo_name="PYDOE_FULLFACT", n_samples=3, eval_jac=True, algo_type="doe"
    )
    with assert_exception(ValueError, snapshot):
        execute_post(problem, post_name="HessianHistory", save=False, show=False)


@pytest.mark.parametrize(
    "use_standardized_objective",
    [True, False],
)
def test_common_scenario(
    use_standardized_objective,
    three_length_common_problem,
    snapshot_matplotlib,
) -> None:
    """Check HessianHistory with objective, standardized or not."""
    three_length_common_problem.use_standardized_objective = use_standardized_objective
    opt = HessianHistory(three_length_common_problem)
    opt.execute(HessianHistory_Settings(save=False))


@pytest.mark.parametrize(
    "case",
    [1, 2],
)
def test_461(case, snapshot_matplotlib) -> None:
    """Check that HessianHistory works with the cases mentioned in issue 461.

    1. Design space of dimension 1 and scalar output.
    2. Design space of dimension > 1 and vector output.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-2, upper_bound=2.0, value=-2.0)
    if case == 2:
        design_space.add_variable("y", lower_bound=-2, upper_bound=2.0, value=-2.0)

    problem = OptimizationProblem(design_space)
    if case == 1:
        problem.objective = ArrayFunction(lambda x: x[0] ** 2, name="func")
    elif case == 2:
        problem.objective = problem.objective = ArrayFunction(
            lambda x: array([x[0] ** 2 + x[1] ** 2]), name="func"
        )
    problem.differentiation_method = problem.ApproximationMode.FINITE_DIFFERENCES

    execute_algo(problem, algo_name="NLOPT_SLSQP", max_iter=5)
    execute_post(problem, post_name="HessianHistory", save=False, show=False)


def test_variable_names(snapshot_matplotlib) -> None:
    execute_post(
        Path(__file__).parent / "mdf_backup.h5",
        post_name="HessianHistory",
        variable_names=["x_2", "x_1"],
        save=False,
        show=False,
    )


def test_no_gradient_history(snapshot) -> None:
    """Check that HessianHistory cannot work without gradient history."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-1, upper_bound=1.0, value=0.5)

    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: x**2, name="f")
    problem.database.store(array([-1]), {"f": array([1])})
    problem.database.store(array([0]), {"f": array([0])})
    problem.database.store(array([1]), {"f": array([1])})
    with assert_exception(ValueError, snapshot):
        execute_post(problem, post_name="HessianHistory", save=False, show=False)
