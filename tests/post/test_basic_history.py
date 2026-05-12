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

from gemseo.post import BasicHistory_Settings
from gemseo.post.basic_history import BasicHistory


@pytest.mark.parametrize(
    ("variable_names", "use_standardized_objective", "options"),
    [
        (["obj", "eq", "neg", "pos", "x"], True, {}),
        (["obj", "eq", "neg", "pos", "x"], False, {}),
        (["obj", "x"], True, {"normalize": True}),
        (["obj", "x"], False, {"normalize": True}),
    ],
)
def test_common_scenario(
    variable_names,
    use_standardized_objective,
    options,
    common_problem,
    snapshot_matplotlib,
) -> None:
    """Check BasicHistory with objective, standardized or not."""
    common_problem.use_standardized_objective = use_standardized_objective
    opt = BasicHistory(common_problem)
    opt.execute(
        BasicHistory_Settings(variable_names=variable_names, save=False, **options)
    )


def test_large_common_scenario(large_common_problem, snapshot_matplotlib) -> None:
    """Check BasicHistory with a common problem and many iterations."""
    opt = BasicHistory(large_common_problem)
    opt.execute(
        BasicHistory_Settings(
            variable_names=["obj", "eq", "neg", "pos", "x"], save=False
        )
    )


@pytest.mark.parametrize(
    "options",
    [
        {"use_best_iteration_history": False},
        {"use_best_iteration_history": True},
    ],
)
def test_use_best_iteration_history(
    options, common_problem_lhs_, snapshot_matplotlib
) -> None:
    """Check the effect of use_best_iteration_history."""
    opt = BasicHistory(common_problem_lhs_)
    opt.execute(
        BasicHistory_Settings(variable_names=["rosen", "x"], save=False, **options)
    )
