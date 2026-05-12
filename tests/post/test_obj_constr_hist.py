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
#       :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest

from gemseo.post import ObjConstrHist_Settings
from gemseo.post.obj_constr_hist import ObjConstrHist


@pytest.mark.parametrize(
    "use_standardized_objective",
    [True, False],
)
def test_common_scenario(
    use_standardized_objective, common_problem, snapshot_matplotlib
) -> None:
    """Check ObjConstrHist."""
    common_problem.use_standardized_objective = use_standardized_objective
    opt = ObjConstrHist(common_problem)
    opt.execute(
        ObjConstrHist_Settings(constraint_names=["eq", "neg", "pos"], save=False)
    )
