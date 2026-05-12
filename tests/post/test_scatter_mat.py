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
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path

import pytest
from numpy import array
from numpy import ones
from numpy import power

from gemseo import create_design_space
from gemseo import create_scenario
from gemseo import execute_post
from gemseo.algos.opt.factory import OPTIMIZATION_LIBRARY_FACTORY
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post import ScatterPlotMatrix_Settings
from gemseo.post.factory import POST_FACTORY
from gemseo.post.scatter_plot_matrix import ScatterPlotMatrix
from gemseo.problems.optimization.power_2 import Power2
from gemseo.utils.testing.helpers import assert_exception

CURRENT_DIR = Path(__file__).parent
POWER2 = Path(__file__).parent / "power2_opt_pb.h5"

pytestmark = pytest.mark.skipif(
    not POST_FACTORY.is_available("ScatterPlotMatrix"),
    reason="ScatterPlotMatrix is not available.",
)


def test_scatter(tmp_wd) -> None:
    """Test the scatter matrix post-processing for all functions.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    problem = Power2()
    OPTIMIZATION_LIBRARY_FACTORY.execute(problem, settings=SLSQP_Settings())
    post = POST_FACTORY.execute(
        problem,
        ScatterPlotMatrix_Settings(
            file_path="scatter1", variable_names=problem.function_names
        ),
    )
    assert len(post.output_file_paths) == 1
    for outf in post.output_file_paths:
        assert Path(outf).exists()


def test_scatter_load(tmp_wd) -> None:
    """Test scatter matrix post-processing with an imported problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    problem = OptimizationProblem.from_hdf(POWER2)
    post = POST_FACTORY.execute(
        problem,
        ScatterPlotMatrix_Settings(
            file_path="scatter2", variable_names=problem.function_names
        ),
    )
    assert len(post.output_file_paths) == 1
    for outf in post.output_file_paths:
        assert Path(outf).exists()

    post = POST_FACTORY.execute(problem, ScatterPlotMatrix_Settings(variable_names=[]))
    for outf in post.output_file_paths:
        assert Path(outf).exists()


def test_non_existent_var(tmp_wd, snapshot) -> None:
    """Test exception when a requested variable does not exist.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    problem = OptimizationProblem.from_hdf(POWER2)
    with assert_exception(ValueError, snapshot):
        POST_FACTORY.execute(
            problem, ScatterPlotMatrix_Settings(variable_names=["foo"])
        )


@pytest.mark.parametrize(
    ("variables", "infile_stem"),
    [
        ([], "empty_list"),
        (["x_shared", "obj"], "subset_2components"),
        (["x_shared", "x_local"], "subset_2variables"),
        (["c_2", "x_shared", "x_local", "obj", "c_1"], "all_var_func"),
    ],
)
def test_scatter_plot(variables, infile_stem, snapshot_matplotlib) -> None:
    """Test images created by the post_process method against references.

    Args:
        variables: The list of variables to be plotted
            in each test case.
        infile_stem: The stem of the hdf5 data file to load.
    """
    infile = CURRENT_DIR / (infile_stem + ".h5")
    execute_post(
        infile,
        post_name="ScatterPlotMatrix",
        save=False,
        file_path="scatter_sellar",
        file_extension="png",
        variable_names=variables,
    )


def test_maximized_func(tmp_wd, sellar_with_2d_array, sellar_disciplines) -> None:
    """Test if the method identifies maximized objectives properly.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    design_space = create_design_space()
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=10.0, value=ones(1))
    design_space.add_variable(
        "x_shared",
        2,
        lower_bound=(-10, 0.0),
        upper_bound=(10.0, 10.0),
        value=array([4.0, 3.0]),
    )
    design_space.add_variable(
        "y_0", lower_bound=-100.0, upper_bound=100.0, value=ones(1)
    )
    design_space.add_variable(
        "y_1", lower_bound=-100.0, upper_bound=100.0, value=ones(1)
    )
    scenario = create_scenario(
        sellar_disciplines,
        "obj",
        design_space,
        formulation_name="MDF",
        maximize_objective=True,
    )
    scenario.add_constraint("c_1", constraint_type=scenario.ConstraintType.INEQ)
    scenario.add_constraint("c_2", constraint_type=scenario.ConstraintType.INEQ)
    scenario.set_differentiation_method("finite_differences")
    scenario.set_algorithm(SLSQP_Settings(max_iter=10))
    scenario.execute()
    post = scenario.post_process(
        ScatterPlotMatrix_Settings(
            save=True,
            file_path="scatter_sellar",
            file_extension="png",
            variable_names=["obj", "x_1", "x_shared"],
        )
    )
    assert len(post.output_file_paths) == 1
    for outf in post.output_file_paths:
        assert outf.exists()


@pytest.mark.parametrize(
    "filter_non_feasible",
    [True, False],
)
def test_filter_non_feasible(filter_non_feasible, snapshot_matplotlib) -> None:
    """Test if the filter_non_feasible option works properly.

    Args:
        filter_non_feasible: If True, remove the non-feasible points from the data.
    """
    # Create a Power2 instance
    problem = Power2()
    # Add feasible points
    problem.database.store(
        array([0.79499653, 0.20792012, 0.96630481]),
        {"pow2": 1.61, "ineq1": -0.0024533, "ineq2": -0.0024533, "eq": -0.00228228},
    )
    problem.database.store(
        array([0.9, 0.9, power(0.9, 1 / 3)]),
        {"pow2": 2.55, "ineq1": -0.229, "ineq2": -0.229, "eq": 0.0},
    )
    # Add two non-feasible points
    problem.database.store(
        array([1.0, 1.0, 0.0]), {"pow2": 2.0, "ineq1": -0.5, "ineq2": -0.5, "eq": 0.9}
    )
    problem.database.store(
        array([0.5, 0.5, 0.5]),
        {"pow2": 0.75, "ineq1": 0.375, "ineq2": 0.375, "eq": 0.775},
    )
    POST_FACTORY.execute(
        problem,
        ScatterPlotMatrix_Settings(
            file_extension="png",
            save=False,
            filter_non_feasible=filter_non_feasible,
            variable_names=["x"],
        ),
    )


def test_filter_non_feasible_exception(snapshot) -> None:
    """Test exception when no feasible points are left after filtering."""
    # Create a Power2 instance
    problem = Power2()
    # Add two non-feasible points
    problem.database.store(
        array([1.0, 1.0, 0.0]), {"pow2": 2.0, "ineq1": -0.5, "ineq2": -0.5, "eq": 0.9}
    )
    problem.database.store(
        array([0.5, 0.5, 0.5]),
        {"pow2": 0.75, "ineq1": 0.375, "ineq2": 0.375, "eq": 0.775},
    )

    with assert_exception(ValueError, snapshot):
        POST_FACTORY.execute(
            problem,
            ScatterPlotMatrix_Settings(filter_non_feasible=True, variable_names=["x"]),
        )


@pytest.mark.parametrize(
    "use_standardized_objective",
    [True, False],
)
def test_common_scenario(
    use_standardized_objective, common_problem, snapshot_matplotlib
) -> None:
    """Check ScatterPlotMatrix with objective, standardized or not."""
    common_problem.use_standardized_objective = use_standardized_objective
    opt = ScatterPlotMatrix(common_problem)
    opt.execute(
        ScatterPlotMatrix_Settings(
            variable_names=["obj", "eq", "neg", "pos", "x"], save=False
        )
    )
