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
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import pytest

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.utils.py23_compat import Path

POWER2 = Path(__file__).parent / "power2_opt_pb.h5"


@pytest.fixture(scope="module")
def factory():
    return PostFactory()


def test_quad_sr1(tmp_wd, factory):
    """Test the quad approx post-processing with the Rosenbrock problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
    """
    problem = Rosenbrock(20)
    OptimizersFactory().execute(problem, "L-BFGS-B")
    post = factory.execute(
        problem,
        "QuadApprox",
        show=False,
        save=True,
        function=problem.objective.name,
        file_path="quadappr",
    )

    for outf in post.output_files:
        assert Path(outf).exists()


def test_quad_sr1_load(tmp_wd, factory):
    """Test the quad approx post-processing from a database.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        factory: Fixture to return a post-processing factory.
    """
    problem = OptimizationProblem.import_hdf(POWER2)
    post = factory.execute(
        problem,
        "QuadApprox",
        show=False,
        save=True,
        function=problem.objective.name,
        file_path="quadappr_sr1",
    )

    for outf in post.output_files:
        assert Path(outf).exists()
