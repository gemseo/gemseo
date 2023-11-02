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
from gemseo import execute_post
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.animation import Animation
from gemseo.problems.topo_opt.topopt_initialize import (
    initialize_design_space_and_discipline_to,
)

DIR_PATH = Path(__file__).parent
POWER2_PATH = DIR_PATH / "power2_opt_pb.h5"


@pytest.mark.parametrize("n_rep", [1, None])
@pytest.mark.parametrize("keep_frames", [True, False])
@pytest.mark.parametrize("tmp_file", ["tmp", "", Path("tmp")])
@pytest.mark.parametrize("first_iteration", [-1, 1])
def test_common_scenario(
    common_problem, n_rep, keep_frames, tmp_file, tmp_wd, first_iteration
):
    """Check Animation with objective, standardized or not."""
    animation = Animation(common_problem)
    output_files = animation.execute(
        opt_post_processor="BasicHistory",
        variable_names=["obj", "eq", "neg", "pos", "x"],
        n_repetitions=n_rep,
        remove_frames=not keep_frames,
        temporary_database_file=tmp_file,
        first_iteration=first_iteration,
    )
    for output_file in output_files:
        assert Path(output_file).exists()


def test_large_common_scenario(large_common_problem, tmp_wd):
    """Check Animation with objective, standardized or not."""
    opt = Animation(large_common_problem)
    output_files = opt.execute(
        variable_names=["obj", "eq", "neg", "pos", "x"],
        opt_post_processor="BasicHistory",
    )
    for output_file in output_files:
        assert Path(output_file).exists()


def test_opt_hist_const(tmp_wd):
    """Test that a problem with constraints is properly rendered."""
    problem = OptimizationProblem.from_hdf(POWER2_PATH)
    output_files = execute_post(
        problem,
        "Animation",
        opt_post_processor="OptHistoryView",
        variable_names=["x"],
        file_path="power2_2",
        obj_min=0.0,
        obj_max=5.0,
    ).output_files
    for output_file in output_files:
        assert Path(output_file).exists()


def test_l_shape(tmp_wd):
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
        formulation="DisciplinaryOpt",
        objective_name="compliance",
        design_space=design_space,
    )
    scenario.add_constraint("volume fraction", "ineq", value=volume_fraction)
    scenario.execute({"max_iter": 10, "algo": "NLOPT_MMA"})
    output_files = scenario.post_process(
        "Animation",
        opt_post_processor="TopologyView",
        n_x=n_x,
        n_y=n_y,
    ).output_files
    for output_file in output_files:
        assert Path(output_file).exists()
