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
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.post.post_factory import PostFactory
from gemseo.post.scatter_mat import ScatterPlotMatrix
from gemseo.problems.analytical.power_2 import Power2
from gemseo.utils.testing import image_comparison
from numpy import array
from numpy import ones
from numpy import power

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"

pytestmark = pytest.mark.skipif(
    not PostFactory().is_available("ScatterPlotMatrix"),
    reason="ScatterPlotMatrix is not available.",
)


def test_scatter(tmp_wd, pyplot_close_all):
    """Test the scatter matrix post-processing for all functions.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
    problem = Power2()
    OptimizersFactory().execute(problem, "SLSQP")
    post = factory.execute(
        problem,
        "ScatterPlotMatrix",
        file_path="scatter1",
        variable_names=problem.get_all_functions_names(),
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()


def test_scatter_load(tmp_wd, pyplot_close_all):
    """Test scatter matrix post-processing with an imported problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
    problem = OptimizationProblem.import_hdf(POWER2)
    post = factory.execute(
        problem,
        "ScatterPlotMatrix",
        file_path="scatter2",
        variable_names=problem.get_all_functions_names(),
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()

    post = factory.execute(problem, "ScatterPlotMatrix", variable_names=[])
    for outf in post.output_files:
        assert Path(outf).exists()


def test_non_existent_var(tmp_wd):
    """Test exception when a requested variable does not exist.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    factory = PostFactory()
    problem = OptimizationProblem.import_hdf(POWER2)
    with pytest.raises(
        ValueError,
        match=r"Cannot build scatter plot matrix: function foo is neither "
        r"among optimization problem functions: .* "
        r"nor design variables: .*",
    ):
        factory.execute(problem, "ScatterPlotMatrix", variable_names=["foo"])


@pytest.mark.parametrize(
    "variables, baseline_images",
    [
        ([], ["empty_list"]),
        (["x_shared", "obj"], ["subset_2components"]),
        (["x_shared", "x_local"], ["subset_2variables"]),
        (["c_2", "x_shared", "x_local", "obj", "c_1"], ["all_var_func"]),
    ],
)
@image_comparison(None)
def test_scatter_plot(baseline_images, variables, pyplot_close_all):
    """Test images created by the post_process method against references.

    Args:
        baseline_images: The reference images to be compared.
        variables: The list of variables to be plotted
            in each test case.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
    design_space = create_design_space()
    design_space.add_variable("x_local", l_b=0.0, u_b=10.0, value=ones(1))
    design_space.add_variable(
        "x_shared", 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=array([4.0, 3.0])
    )
    design_space.add_variable("y_0", l_b=-100.0, u_b=100.0, value=ones(1))
    design_space.add_variable("y_1", l_b=-100.0, u_b=100.0, value=ones(1))
    scenario = create_scenario(
        disciplines, "MDF", objective_name="obj", design_space=design_space
    )
    scenario.add_constraint("c_1", "ineq")
    scenario.add_constraint("c_2", "ineq")
    scenario.set_differentiation_method("finite_differences")
    scenario.default_inputs = {"max_iter": 10, "algo": "SLSQP"}
    scenario.execute()
    post = scenario.post_process(
        "ScatterPlotMatrix",
        save=False,
        file_path="scatter_sellar",
        file_extension="png",
        variable_names=variables,
    )
    post.figures


def test_maximized_func(tmp_wd, pyplot_close_all):
    """Test if the method identifies maximized objectives properly.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
    design_space = create_design_space()
    design_space.add_variable("x_local", l_b=0.0, u_b=10.0, value=ones(1))
    design_space.add_variable(
        "x_shared", 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=array([4.0, 3.0])
    )
    design_space.add_variable("y_0", l_b=-100.0, u_b=100.0, value=ones(1))
    design_space.add_variable("y_1", l_b=-100.0, u_b=100.0, value=ones(1))
    scenario = create_scenario(
        disciplines,
        "MDF",
        objective_name="obj",
        design_space=design_space,
        maximize_objective=True,
    )
    scenario.add_constraint("c_1", "ineq")
    scenario.add_constraint("c_2", "ineq")
    scenario.set_differentiation_method("finite_differences")
    scenario.default_inputs = {"max_iter": 10, "algo": "SLSQP"}
    scenario.execute()
    post = scenario.post_process(
        "ScatterPlotMatrix",
        save=True,
        file_path="scatter_sellar",
        file_extension="png",
        variable_names=["obj", "x_local", "x_shared"],
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()


@pytest.mark.parametrize(
    "filter_non_feasible, baseline_images",
    [(True, ["power_2_filtered"]), (False, ["power_2_not_filtered"])],
)
@image_comparison(None)
def test_filter_non_feasible(filter_non_feasible, baseline_images, pyplot_close_all):
    """Test if the filter_non_feasible option works properly.

    Args:
        filter_non_feasible: If True, remove the non-feasible points from the data.
        baseline_images: The reference images to be compared.
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
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
    post = factory.execute(
        problem,
        "ScatterPlotMatrix",
        file_extension="png",
        save=False,
        filter_non_feasible=filter_non_feasible,
        variable_names=["x"],
    )
    post.figures


def test_filter_non_feasible_exception():
    """Test exception when no feasible points are left after filtering."""
    factory = PostFactory()
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

    with pytest.raises(ValueError, match="No feasible points were found."):
        factory.execute(
            problem, "ScatterPlotMatrix", filter_non_feasible=True, variable_names=["x"]
        )


TEST_PARAMETERS = {
    "standardized": (True, ["ScatterPlotMatrix_standardized"]),
    "unstandardized": (False, ["ScatterPlotMatrix_unstandardized"]),
}


@pytest.mark.parametrize(
    "use_standardized_objective, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective, baseline_images, common_problem, pyplot_close_all
):
    """Check ScatterPlotMatrix with objective, standardized or not."""
    opt = ScatterPlotMatrix(common_problem)
    common_problem.use_standardized_objective = use_standardized_objective
    opt.execute(variable_names=["obj", "eq", "neg", "pos", "x"], save=False)
