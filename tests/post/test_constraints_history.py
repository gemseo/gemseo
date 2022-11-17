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
from gemseo.post.constraints_history import ConstraintsHistory
from gemseo.utils.testing import image_comparison


def test_function_error(common_problem):
    """Test a ValueError is raised for a non-existent function."""
    with pytest.raises(
        ValueError,
        match="Cannot build constraints history plot, "
        "function foo is not among the constraints names "
        "or does not exist.",
    ):
        ConstraintsHistory(common_problem).execute(save=False, constraint_names=["foo"])


TEST_PARAMETERS = {"default": ["ConstraintsHistory_default"]}


@pytest.mark.parametrize(
    "baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(baseline_images, common_problem, pyplot_close_all):
    """Check ConstraintsHistory."""
    opt = ConstraintsHistory(common_problem)
    opt.execute(constraint_names=["eq", "neg", "pos"], save=False)
