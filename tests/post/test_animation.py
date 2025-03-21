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
from __future__ import annotations

from pathlib import Path

import pytest

from gemseo import create_scenario
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post.animation import Animation
from gemseo.post.factory import PostFactory
from gemseo.problems.topology_optimization.topopt_initialize import (
    initialize_design_space_and_discipline_to,
)

DIR_PATH = Path(__file__).parent
POWER2_PATH = DIR_PATH / "power2_opt_pb.h5"


@pytest.mark.parametrize("n_rep", [1, 0])
@pytest.mark.parametrize("keep_frames", [True, False])
@pytest.mark.parametrize("tmp_file", ["tmp", "", Path("tmp")])
@pytest.mark.parametrize("first_iteration", [-1, 1])
def test_common_scenario(
    common_problem, n_rep, keep_frames, tmp_file, tmp_wd, first_iteration
) -> None:
    """Check Animation with objective, standardized or not."""
    animation = Animation(common_problem)
    post_processing = PostFactory().create("BasicHistory", common_problem)
    pp_settings = post_processing.Settings(
        variable_names=["obj", "eq", "neg", "pos", "x"],
    )
    output_files = animation.execute(
        n_repetitions=n_rep,
        remove_frames=not keep_frames,
        temporary_database_path=tmp_file,
        first_iteration=first_iteration,
        post_processing=post_processing,
        post_processing_settings=pp_settings,
    )
    for output_file in output_files:
        assert Path(output_file).exists()


@pytest.mark.slow
def test_large_common_scenario(large_common_problem, tmp_wd) -> None:
    """Check Animation with objective, standardized or not."""
    opt = Animation(large_common_problem)
    post_processing = PostFactory().create("BasicHistory", large_common_problem)
    pp_settings = post_processing.Settings(
        variable_names=["obj", "eq", "neg", "pos", "x"],
    )
    output_files = opt.execute(
        post_processing=post_processing,
        post_processing_settings=pp_settings,
    )
    for output_file in output_files:
        assert Path(output_file).exists()


def test_opt_hist_const(tmp_wd) -> None:
    """Test that a problem with constraints is properly rendered."""
    problem = OptimizationProblem.from_hdf(POWER2_PATH)
    opt = Animation(problem)
    post_processing = PostFactory().create("OptHistoryView", problem)
    pp_settings = post_processing.Settings(
        variable_names=["x"],
        file_path="power2_2",
        obj_min=0.0,
        obj_max=5.0,
    )
    output_files = opt.execute(
        post_processing=post_processing,
        post_processing_settings=pp_settings,
    )
    for output_file in output_files:
        assert Path(output_file).exists()


def test_l_shape(tmp_wd) -> None:
    """Test the plot of the solution of the L-shape topology optimization."""
    volume_fraction = 0.3
    problem_name = "L-Shape"
    n_x = 25
    n_y = 25
    e0 = 1
    nu = 0.3
    penalty = 3
    min_member_size = 1.5
    design_space, disciplines = initialize_design_space_and_discipline_to(
        problem=problem_name,
        n_x=n_x,
        n_y=n_y,
        e0=e0,
        nu=nu,
        penalty=penalty,
        min_member_size=min_member_size,
        vf0=volume_fraction,
    )
    scenario = create_scenario(
        disciplines,
        "compliance",
        design_space,
        formulation_name="DisciplinaryOpt",
    )
    scenario.add_constraint(
        "volume fraction", constraint_type="ineq", value=volume_fraction
    )
    scenario.execute(algo_name="NLOPT_MMA", max_iter=10)
    post_processing = PostFactory().create(
        "TopologyView", scenario.formulation.optimization_problem
    )
    pp_settings = post_processing.Settings(n_x=n_x, n_y=n_y)
    output_files = scenario.post_process(
        post_name="Animation",
        post_processing=post_processing,
        post_processing_settings=pp_settings,
    ).output_file_paths
    for output_file in output_files:
        assert output_file.exists()
