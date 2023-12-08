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

from __future__ import annotations

import pytest

from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline


@pytest.fixture(params=[0.0, 0.1, 1.0])
def x0(request):
    return request.param


@pytest.fixture(params=[0.0, 0.1, 1.0])
def y0(request):
    return request.param


@pytest.fixture()
def analytical_test_2d_ineq(x0, y0):
    """Test for lagrange multiplier."""
    disc = AnalyticDiscipline(
        name="2D_test", expressions={"f": "(x-1)**2+(y-1)**2", "g": "x+y-1"}
    )
    ds = DesignSpace()
    ds.add_variable("x", l_b=0.0, u_b=1.0, value=x0)
    ds.add_variable("y", l_b=0.0, u_b=1.0, value=y0)
    scenario = create_scenario(
        disciplines=[disc],
        formulation="DisciplinaryOpt",
        objective_name="f",
        design_space=ds,
    )
    scenario.add_constraint("g", "ineq")
    return scenario


@pytest.fixture()
def analytical_test_2d_eq(x0, y0):
    """Test for lagrange multiplier."""
    disc = AnalyticDiscipline(
        name="2D_test", expressions={"f": "(x)**2+(y)**2", "g": "x+y-1"}
    )
    ds = DesignSpace()
    ds.add_variable("x", l_b=0.0, u_b=1.0, value=x0)
    ds.add_variable("y", l_b=0.0, u_b=1.0, value=y0)
    scenario = create_scenario(
        disciplines=[disc],
        formulation="DisciplinaryOpt",
        objective_name="f",
        design_space=ds,
    )
    scenario.add_constraint("g", "eq")
    return scenario


@pytest.fixture()
def analytical_test_2d__multiple_eq():
    """Test for lagrange multiplier."""
    x0 = 4.0
    disc = AnalyticDiscipline(
        name="2D_test",
        expressions={"f": "(x)**2+(y)**2+(z)**2", "h1": "x-y", "h2": "y-z"},
    )
    ds = DesignSpace()
    ds.add_variable("x", l_b=0.0, u_b=4.0, value=x0)
    ds.add_variable("y", l_b=1.0, u_b=4.0, value=x0)
    ds.add_variable("z", l_b=2.0, u_b=4.0, value=x0)
    scenario = create_scenario(
        disciplines=[disc],
        formulation="DisciplinaryOpt",
        objective_name="f",
        design_space=ds,
    )
    scenario.add_constraint("h1", "eq")
    scenario.add_constraint("h2", "eq")
    return scenario


@pytest.fixture()
def analytical_test_2d_mixed_rank_deficient():
    """Test for lagrange multiplier."""
    disc = AnalyticDiscipline(
        name="2D_test",
        expressions={
            "f": "x**2+y**2+z**2",
            "g": "x+y+z-1",
            "h": "(x-1.)**2+(y-1)**2+(z-1)**2-4./3.",
        },
    )
    ds = DesignSpace()
    ds.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    ds.add_variable("y", l_b=0.0, u_b=1.0, value=0.5)
    ds.add_variable("z", l_b=0.0, u_b=1.0, value=0.5)
    scenario = create_scenario(
        disciplines=[disc],
        formulation="DisciplinaryOpt",
        objective_name="f",
        design_space=ds,
    )
    scenario.add_constraint("g", "ineq")
    scenario.add_constraint("h", "eq")
    return scenario
