# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import numpy as np
import pytest

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.power_2 import Power2
from gemseo.utils.py23_compat import Path

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"


@pytest.fixture(scope="module")
def problem():
    problem = Power2()
    problem.x_0 = np.ones(3) * 50
    problem.l_bounds = -np.ones(3)
    problem.u_bounds = np.ones(3) * 50
    OptimizersFactory().execute(problem, "SLSQP")
    return problem


@pytest.fixture(scope="module")
def problem_history():
    return OptimizationProblem.import_hdf(file_path=POWER2)


@pytest.fixture(scope="module")
def factory():
    return PostFactory()


def test_objconstr(tmp_wd, factory, problem):
    """Test the objective constraint history post-processing with the Power2 problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        problem: Fixture to return a Power2 `OptimizationProblem` after execution.
        factory: Fixture to return a post-processing factory.
    """
    post = factory.execute(
        problem,
        "ObjConstrHist",
        save=True,
        show=False,
        file_path="ObjConstrHist1",
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()


def test_objconstr_load(tmp_wd, problem_history, factory):
    """Test the objective constraint history post-processing from an hdf5 database.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        problem_history: Fixture to return a Power2 `OptimizationProblem` from an
            hdf5 database.
        factory: Fixture to return a post-processing factory.
    """
    post = factory.execute(
        problem_history,
        "ObjConstrHist",
        save=True,
        show=False,
        file_path="ObjConstrHist2",
        constr_names=["ineq1", "ineq2"],
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()


def test_objconstr_name(tmp_wd, problem, factory):
    """Test the constraint filter with the Power2 problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        problem: Fixture to return a Power2 `OptimizationProblem` after execution.
        factory: Fixture to return a post-processing factory.
    """
    post = factory.execute(
        problem,
        "ObjConstrHist",
        file_path="ObjConstrHist3",
        save=True,
        show=False,
        constr_names=["eq"],
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()


def test_objconstr_name_load(tmp_wd, problem_history, factory):
    """Test the constraint filter with an hdf5 database.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        problem_history: Fixture to return a Power2 `OptimizationProblem` from an
            hdf5 database.
        factory: Fixture to return a post-processing factory.
    """
    post = factory.execute(
        problem_history,
        "ObjConstrHist",
        save=True,
        show=False,
        constr_names=["ineq1", "ineq2"],
        file_path="ObjConstrHist4",
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()
