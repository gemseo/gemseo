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


@pytest.fixture
def analytical_test_2d_ineq(x0, y0):
    """Test for lagrange multiplier."""
    disc = AnalyticDiscipline({"f": "(x-1)**2+(y-1)**2", "g": "x+y-1"}, name="2D_test")
    ds = DesignSpace()
    ds.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=x0)
    ds.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=y0)
    scenario = create_scenario(
        [disc],
        "f",
        ds,
        formulation_name="DisciplinaryOpt",
    )
    scenario.add_constraint("g", constraint_type="ineq")
    return scenario


@pytest.fixture
def analytical_test_2d_eq(x0, y0):
    """Test for lagrange multiplier."""
    disc = AnalyticDiscipline({"f": "(x)**2+(y)**2", "g": "x+y-1"}, name="2D_test")
    ds = DesignSpace()
    ds.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=x0)
    ds.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=y0)
    scenario = create_scenario(
        [disc],
        "f",
        ds,
        formulation_name="DisciplinaryOpt",
    )
    scenario.add_constraint("g")
    return scenario


@pytest.fixture
def analytical_test_2d__multiple_eq():
    """Test for lagrange multiplier."""
    x0 = 4.0
    disc = AnalyticDiscipline(
        {"f": "(x)**2+(y)**2+(z)**2", "h1": "x-y", "h2": "y-z"},
        name="2D_test",
    )
    ds = DesignSpace()
    ds.add_variable("x", lower_bound=0.0, upper_bound=4.0, value=x0)
    ds.add_variable("y", lower_bound=1.0, upper_bound=4.0, value=x0)
    ds.add_variable("z", lower_bound=2.0, upper_bound=4.0, value=x0)
    scenario = create_scenario(
        [disc],
        "f",
        ds,
        formulation_name="DisciplinaryOpt",
    )
    scenario.add_constraint("h1")
    scenario.add_constraint("h2")
    return scenario


@pytest.fixture
def analytical_test_2d_mixed_rank_deficient():
    """Test for lagrange multiplier."""
    disc = AnalyticDiscipline(
        {
            "f": "x**2+y**2+z**2",
            "g": "x+y+z-1",
            "h": "(x-1.)**2+(y-1)**2+(z-1)**2-4./3.",
        },
        name="2D_test",
    )
    ds = DesignSpace()
    ds.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    ds.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=0.5)
    ds.add_variable("z", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = create_scenario(
        [disc],
        "f",
        ds,
        formulation_name="DisciplinaryOpt",
    )
    scenario.add_constraint("g", constraint_type="ineq")
    scenario.add_constraint("h")
    return scenario
