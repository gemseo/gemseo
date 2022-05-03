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
from pathlib import Path

import pytest
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory

pytestmark = pytest.mark.skipif(
    not PostFactory().is_available("SOM"),
    reason="SOM plot is not available.",
)

POWER2_PATH = Path(__file__).parent / "power2_opt_pb.h5"


def test_som(tmp_wd):
    """Test the SOM post processing with the Power2 problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    problem = OptimizationProblem.import_hdf(POWER2_PATH)
    factory = PostFactory()
    for val in problem.database.values():
        val.pop("pow2")
    post = factory.execute(problem, "SOM", n_x=4, n_y=3, show=False, save=True)
    assert len(post.output_files) == 1
    assert Path(post.output_files[0]).exists()


def test_som_annotate(tmp_wd):
    """Test the annotate option of the post processor.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    problem = OptimizationProblem.import_hdf(POWER2_PATH)
    factory = PostFactory()
    for val in problem.database.values():
        val.pop("pow2")
    post = factory.execute(
        problem, "SOM", n_x=4, n_y=3, show=False, save=True, annotate=True
    )
    assert len(post.output_files) == 1
    assert Path(post.output_files[0]).exists()
