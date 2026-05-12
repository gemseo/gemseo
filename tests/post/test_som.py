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

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post import SOM_Settings
from gemseo.post.factory import POST_FACTORY

pytestmark = pytest.mark.skipif(
    not POST_FACTORY.is_available("SOM"),
    reason="SOM plot is not available.",
)

POWER2_PATH = Path(__file__).parent / "power2_opt_pb.h5"
SELLAR_PATH = Path(__file__).parent / "modified_sellar_opt_pb.h5"
SOBIESKI_PATH = Path(__file__).parent / "sobieski_all_gradients.h5"


@pytest.mark.parametrize(
    ("is_annotated", "h5_path"),
    [
        (True, POWER2_PATH),
        (False, POWER2_PATH),
        (True, SELLAR_PATH),
        (False, SELLAR_PATH),
        (True, SOBIESKI_PATH),
        (False, SOBIESKI_PATH),
    ],
)
def test_som(is_annotated, h5_path, snapshot_matplotlib) -> None:
    """Test the SOM post-processing."""
    problem = OptimizationProblem.from_hdf(h5_path)
    POST_FACTORY.execute(
        problem, SOM_Settings(n_x=4, n_y=3, save=False, annotate=is_annotated)
    )
