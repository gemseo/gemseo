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

import re

import pytest
from gemseo.post.para_coord import ParallelCoordinates
from gemseo.utils.testing import image_comparison
from numpy import array


TEST_PARAMETERS = {
    "standardized": (True, ["PC_standardized_0", "PC_standardized_1"]),
    "unstandardized": (False, ["PC_unstandardized_0", "PC_unstandardized_1"]),
}


@pytest.mark.parametrize(
    "use_standardized_objective, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective, baseline_images, common_problem, pyplot_close_all
):
    """Check ParallelCoordinates with objective, standardized or not."""
    opt = ParallelCoordinates(common_problem)
    common_problem.use_standardized_objective = use_standardized_objective
    opt.execute(save=False)


def test_shape_error():
    """Check the error raised by parallel_coordinates if shapes are inconsistent."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The data shape (1, 1) is not equal to the expected one (2, 1)."
        ),
    ):
        ParallelCoordinates.parallel_coordinates(array([[1]]), ["x"], [0.0, 0.5])
