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

from gemseo.algos.opt.factory import OPTIMIZATION_LIBRARY_FACTORY
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.post import QuadApprox_Settings
from gemseo.post.quad_approx import QuadApprox
from gemseo.problems.optimization.power_2 import Power2


@pytest.mark.parametrize(
    ("use_standardized_objective", "function"),
    [
        (True, "rosen"),
        (False, "rosen"),
    ],
)
def test_common_scenario(
    use_standardized_objective, function, common_problem_, snapshot_matplotlib
) -> None:
    """Check QuadApprox with objective, standardized or not."""
    common_problem_.use_standardized_objective = use_standardized_objective
    opt = QuadApprox(common_problem_)
    opt.execute(QuadApprox_Settings(function=function, save=False))


def test_function_not_in_constraints():
    """Tests QuadApprox when the passed function is not part of the constraints."""
    problem = Power2()
    problem.use_standardized_objective = True
    OPTIMIZATION_LIBRARY_FACTORY.execute(problem, settings=SLSQP_Settings(max_iter=5))
    opt = QuadApprox(problem)
    opt.execute(QuadApprox_Settings(function="ineq1", save=False))
