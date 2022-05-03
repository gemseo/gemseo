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
from pathlib import Path

import pytest
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"


@pytest.fixture(scope="module")
def problem():
    return OptimizationProblem.import_hdf(file_path=POWER2)


@pytest.fixture(scope="module")
def factory():
    return PostFactory()


def test_constraints_history(tmp_wd, factory):
    """Test the constraints history post-processing with the Power2 problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
    """
    problem = Power2()
    OptimizersFactory().execute(problem, "SLSQP")
    post = factory.execute(
        problem,
        "ConstraintsHistory",
        file_path="lines_chart1",
        save=True,
        show=False,
        constraints_list=problem.get_constraints_names(),
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()


def test_constraints_history_load(tmp_wd, problem, factory):
    """Test the radar chart post processing from a database.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        problem: Fixture to return a Power2 `OptimizationProblem` from an hdf5 database.
        factory: Fixture to return a post-processing factory.
    """
    post = factory.execute(
        problem,
        "ConstraintsHistory",
        save=True,
        show=False,
        constraints_list=problem.get_constraints_names(),
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()


def test_function_error(tmp_wd, problem, factory):
    """Test a ValueError is raised for a non-existent function.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        problem: Fixture to return a Power2 `OptimizationProblem` from an hdf5 database.
        factory: Fixture to return a post-processing factory.
    """
    with pytest.raises(
        ValueError,
        match="Cannot build constraints history plot, "
        "function toto is not among the constraints names "
        "or does not exist.",
    ):
        factory.execute(
            problem,
            "ConstraintsHistory",
            save=True,
            show=False,
            constraints_list=["toto"],
        )
