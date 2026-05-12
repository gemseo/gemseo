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

from pathlib import Path

import matplotlib
import pytest
from packaging import version

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post import ConstraintRadar_Settings
from gemseo.post.constraint_radar import ConstraintRadar
from gemseo.utils.testing.helpers import assert_exception

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"


@pytest.fixture(scope="module")
def problem():
    return OptimizationProblem.from_hdf(file_path=POWER2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"iteration": None},
        {"iteration": -2},
        {"iteration": 2},
        {"iteration": 2, "constraint_names": ["ineq1", "eq"]},
        {"show_names_radially": True},
    ],
)
def test_post(kwargs, problem, snapshot_matplotlib) -> None:
    """Test the radar chart post-processing with the Power2 problem."""
    post = ConstraintRadar(problem)
    post.execute(ConstraintRadar_Settings(save=False, show=False, **kwargs))


def test_function_error(problem, snapshot) -> None:
    """Test a ValueError is raised for a non-existent function."""
    post = ConstraintRadar(problem)
    with assert_exception(ValueError, snapshot):
        post.execute(ConstraintRadar_Settings(save=False, constraint_names=["foo"]))


def test_iteration_error(problem, snapshot) -> None:
    """Test a ValueError is raised with ill-defined iteration."""
    len(problem.database)
    post = ConstraintRadar(problem)
    with assert_exception(ValueError, snapshot):
        post.execute(
            ConstraintRadar_Settings(
                save=False,
                constraint_names=problem.constraints.get_names(),
                iteration=1000,
            )
        )


@pytest.mark.skipif(
    version.parse(matplotlib.__version__) < version.parse("3.10.0"),
    reason="Does not work with matplotlib < 3.10.0",
)
def test_common_scenario(common_problem, snapshot_matplotlib) -> None:
    """Check ConstraintRadar."""
    opt = ConstraintRadar(common_problem)
    opt.execute(
        ConstraintRadar_Settings(constraint_names=["eq", "neg", "pos"], save=False)
    )
