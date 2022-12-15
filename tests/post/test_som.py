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
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.post_factory import PostFactory
from gemseo.utils.testing import image_comparison

pytestmark = pytest.mark.skipif(
    not PostFactory().is_available("SOM"),
    reason="SOM plot is not available.",
)

POWER2_PATH = Path(__file__).parent / "power2_opt_pb.h5"
SELLAR_PATH = Path(__file__).parent / "modified_sellar_opt_pb.h5"

TEST_PARAMETERS = {
    "SOM_Power2_annotated": (True, POWER2_PATH, ["SOM_Power2_annotated"]),
    "SOM_Power2_not_annotated": (False, POWER2_PATH, ["SOM_Power2_not_annotated"]),
    "SOM_Sellar_annotated": (False, SELLAR_PATH, ["SOM_Sellar_not_annotated"]),
    "SOM_Sellar_not_annotated": (False, SELLAR_PATH, ["SOM_Sellar_annotated"]),
}


@pytest.mark.parametrize(
    "is_annotated, h5_path, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_som(is_annotated, h5_path, baseline_images, pyplot_close_all):
    """Test the SOM post-processing."""
    problem = OptimizationProblem.import_hdf(h5_path)
    PostFactory().execute(
        problem, "SOM", n_x=4, n_y=3, save=False, annotate=is_annotated
    )
