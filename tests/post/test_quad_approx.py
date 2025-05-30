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

from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.post.quad_approx import QuadApprox
from gemseo.problems.optimization.power_2 import Power2
from gemseo.utils.testing.helpers import image_comparison

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


@pytest.mark.parametrize(
    ("use_standardized_objective", "function", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective, function, baseline_images, common_problem_
) -> None:
    """Check QuadApprox with objective, standardized or not."""
    common_problem_.use_standardized_objective = use_standardized_objective
    opt = QuadApprox(common_problem_)
    opt.execute(function=function, save=False)


def test_function_not_in_constraints():
    """Tests QuadApprox when the passed function is not part of the constraints."""
    problem = Power2()
    problem.use_standardized_objective = True
    OptimizationLibraryFactory().execute(problem, algo_name="SLSQP", max_iter=5)
    opt = QuadApprox(problem)
    opt.execute(function="ineq1", save=False)
