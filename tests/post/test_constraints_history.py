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
#       :author: Pierre-Jean Barjhoux
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest

from gemseo.post import ConstraintsHistory_Settings
from gemseo.post.constraints_history import ConstraintsHistory
from gemseo.utils.testing.helpers import assert_exception


def test_function_error(common_problem, snapshot) -> None:
    """Test a ValueError is raised for a non-existent function."""
    with assert_exception(ValueError, snapshot):
        ConstraintsHistory(common_problem).execute(
            ConstraintsHistory_Settings(save=False, constraint_names=["foo"])
        )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"line_style": ""},
        {"line_style": "-"},
        {"add_points": False},
    ],
)
def test_common_scenario(options, common_problem, snapshot_matplotlib) -> None:
    """Check ConstraintsHistory."""
    post = ConstraintsHistory(common_problem)
    post.execute(
        ConstraintsHistory_Settings(
            constraint_names=["eq", "neg", "pos"], save=False, **options
        )
    )
