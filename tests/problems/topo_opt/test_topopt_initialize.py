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
#        :author: Simone Coniglio
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.problems.topo_opt.topopt_initialize import (
    initialize_design_space_and_discipline_to,
)


@pytest.mark.parametrize("problem", ["MBB", "L-Shape", "Short_Cantilever"])
def test_initialize_design_space_and_discipline_to(problem):
    """"""
    ds, disciplines = initialize_design_space_and_discipline_to(
        problem=problem,
        n_x=10,
        n_y=10,
        e0=1,
        nu=0.3,
        penalty=3,
        min_member_size=1.5,
        vf0=0.3,
    )
    assert isinstance(ds, DesignSpace)
    assert isinstance(disciplines, list)
    assert all(isinstance(disc, MDODiscipline) for disc in disciplines)


def test_not_implemented():
    with pytest.raises(NotImplementedError):
        ds, disciplines = initialize_design_space_and_discipline_to(
            problem="test",
            n_x=10,
            n_y=10,
            e0=1,
            nu=0.3,
            penalty=3,
            min_member_size=1.5,
            vf0=0.3,
        )
