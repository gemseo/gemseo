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
#       :author: Pierre-Jean Barjhoux
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re
from pathlib import Path

import pytest
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.radar_chart import RadarChart
from gemseo.utils.testing import image_comparison

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"


@pytest.fixture(scope="module")
def problem():
    return OptimizationProblem.import_hdf(file_path=POWER2)


TEST_PARAMETERS = {
    "default": ({}, ["RadarChart_default"]),
    "opt": ({"iteration": "opt"}, ["RadarChart_opt"]),
    "negative": ({"iteration": -2}, ["RadarChart_negative"]),
    "positive": ({"iteration": 2}, ["RadarChart_positive"]),
    "names": (
        {"iteration": 2, "constraint_names": ["ineq1", "eq"]},
        ["RadarChart_names"],
    ),
    "show_names_radially": ({"show_names_radially": True}, ["RadarChart_radial"]),
}


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_post(kwargs, baseline_images, problem, pyplot_close_all):
    """Test the radar chart post-processing with the Power2 problem."""
    post = RadarChart(problem)
    post.execute(save=False, show=False, **kwargs)


def test_function_error(problem):
    """Test a ValueError is raised for a non-existent function."""
    post = RadarChart(problem)
    with pytest.raises(
        ValueError,
        match=(
            r"The names \[.?'foo'\] are not names of constraints "
            r"stored in the database\."
        ),
    ):
        post.execute(save=False, constraint_names=["foo"])


def test_iteration_error(problem):
    """Test a ValueError is raised with ill defined iteration."""
    n_iterations = len(problem.database)
    post = RadarChart(problem)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The requested iteration 1000 is neither in ({},...,0,...,{}) "
            "nor equal to the tag {}.".format(
                -n_iterations + 1, n_iterations - 1, RadarChart.OPTIMUM
            )
        ),
    ):
        post.execute(
            save=False, constraint_names=problem.get_constraints_names(), iteration=1000
        )


TEST_PARAMETERS = {"default": ["RadarChart_common_problem"]}


@pytest.mark.parametrize(
    "baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(baseline_images, common_problem, pyplot_close_all):
    """Check RadarChart."""
    opt = RadarChart(common_problem)
    opt.execute(constraint_names=["eq", "neg", "pos"], save=False)
