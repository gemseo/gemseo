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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest

from gemseo.post.basic_history import BasicHistory
from gemseo.utils.testing.helpers import image_comparison


@pytest.mark.parametrize(
    ("variable_names", "use_standardized_objective", "options", "baseline_images"),
    [
        (["obj", "eq", "neg", "pos", "x"], True, {}, ["BasicHistory_standardized"]),
        (["obj", "eq", "neg", "pos", "x"], False, {}, ["BasicHistory_unstandardized"]),
        (
            ["obj", "x"],
            True,
            {"normalize": True},
            ["BasicHistory_standardized_normalize"],
        ),
        (
            ["obj", "x"],
            False,
            {"normalize": True},
            ["BasicHistory_unstandardized_normalize"],
        ),
    ],
)
@image_comparison(None)
def test_common_scenario(
    variable_names,
    use_standardized_objective,
    options,
    baseline_images,
    common_problem,
    pyplot_close_all,
):
    """Check BasicHistory with objective, standardized or not."""
    opt = BasicHistory(common_problem)
    common_problem.use_standardized_objective = use_standardized_objective
    opt.execute(variable_names=variable_names, save=False, **options)


@pytest.mark.parametrize("baseline_images", [("BasicHistory_many_iterations",)])
@image_comparison(None)
def test_large_common_scenario(baseline_images, large_common_problem, pyplot_close_all):
    """Check BasicHistory with a common problem and many iterations."""
    opt = BasicHistory(large_common_problem)
    opt.execute(variable_names=["obj", "eq", "neg", "pos", "x"], save=False)
