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
from gemseo.problems.sobieski.design_space import create_design_space
from gemseo.problems.sobieski.disciplines import create_disciplines

DISCIPLINES = [
    "SobieskiStructure",
    "SobieskiAerodynamics",
    "SobieskiPropulsion",
    "SobieskiMission",
]


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_create_disciplines(dtype):
    """Check the creation of the disciplines."""
    disciplines = create_disciplines(dtype)
    assert [discipline.name for discipline in disciplines] == DISCIPLINES

    for discipline in disciplines:
        assert str(discipline.default_inputs["x_shared"].dtype) == dtype


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_create_design_space(dtype):
    """Check the creation of the design space."""
    design_space = create_design_space(dtype)
    assert "x_shared" in design_space
    assert str(design_space.get_current_value(["x_shared"]).dtype) == dtype
