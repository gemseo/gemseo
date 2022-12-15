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

import sys

import pytest
from gemseo.post.quad_approx import QuadApprox
from gemseo.utils.testing import image_comparison


TEST_PARAMETERS = {
    "standardized": (
        True,
        "rosen",
        ["QuadApprox_standardized_0", "QuadApprox_standardized_1"],
    ),
    "unstandardized": (
        False,
        "rosen",
        ["QuadApprox_unstandardized_0", "QuadApprox_unstandardized_1"],
    ),
}


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python 3.8 or greater")
@pytest.mark.parametrize(
    "use_standardized_objective, function, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective,
    function,
    baseline_images,
    common_problem_,
    pyplot_close_all,
):
    """Check QuadApprox with objective, standardized or not."""
    opt = QuadApprox(common_problem_)
    common_problem_.use_standardized_objective = use_standardized_objective
    opt.execute(function=function, save=False)
