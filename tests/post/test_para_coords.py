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

import pytest
from numpy import array

from gemseo.post import ParallelCoordinates_Settings
from gemseo.post.parallel_coordinates import ParallelCoordinates
from gemseo.utils.testing.helpers import assert_exception


@pytest.mark.parametrize(
    "use_standardized_objective",
    [True, False],
)
def test_common_scenario(
    use_standardized_objective, common_problem, snapshot_matplotlib
) -> None:
    """Check ParallelCoordinates with objective, standardized or not."""
    common_problem.use_standardized_objective = use_standardized_objective
    opt = ParallelCoordinates(common_problem)
    opt.execute(ParallelCoordinates_Settings(save=False))


def test_shape_error(snapshot) -> None:
    """Check the error raised by parallel_coordinates if shapes are inconsistent."""
    with assert_exception(ValueError, snapshot):
        ParallelCoordinates._ParallelCoordinates__parallel_coordinates(
            array([[1]]), ["x"], [0.0, 0.5], (0.0, 0.0)
        )
