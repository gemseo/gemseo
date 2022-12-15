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
from gemseo.post.robustness import Robustness
from gemseo.utils.testing import image_comparison

TEST_PARAMETERS = {
    "standardized": (True, ["Robustness_standardized"]),
    "unstandardized": (False, ["Robustness_unstandardized"]),
}


@pytest.mark.parametrize(
    "use_standardized_objective, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective,
    baseline_images,
    common_problem,
    pyplot_close_all,
):
    """Check Robustness with objective, standardized or not."""
    opt = Robustness(common_problem)
    common_problem.use_standardized_objective = use_standardized_objective
    opt.execute(save=False)


@pytest.mark.parametrize(
    "baseline_images",
    (["Robustness_stddev"],),
    indirect=["baseline_images"],
    ids=["stddev"],
)
@image_comparison(None)
def test_common_scenario_std(
    baseline_images,
    common_problem,
    pyplot_close_all,
):
    """Check Robustness with a custom standard deviation."""
    opt = Robustness(common_problem)
    opt.execute(stddev=0.2, save=False)
