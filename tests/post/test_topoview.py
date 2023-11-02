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
from gemseo.problems.topo_opt.topopt_initialize import (
    initialize_design_space_and_discipline_to,
)
from gemseo.utils.testing.helpers import image_comparison


@pytest.fixture(scope="module")
def scenario_and_dimensions():
    """Scenario and dimensions of the L-shape problem."""
    vf0 = 0.3
    n_el = 25
    design_space, disciplines = initialize_design_space_and_discipline_to(
        problem="L-Shape",
        n_x=n_el,
        n_y=n_el,
        e0=1,
        nu=0.3,
        penalty=3,
        min_member_size=1.5,
        vf0=vf0,
    )
    scenario = create_scenario(
        disciplines,
        formulation="DisciplinaryOpt",
        objective_name="compliance",
        design_space=design_space,
    )
    scenario.add_observable("xPhys")
    scenario.add_constraint("volume fraction", "ineq", value=vf0)
    scenario.execute({"max_iter": 1, "algo": "NLOPT_MMA"})
    return scenario, n_el, n_el


@image_comparison(["l_shape_solution"])
def test_l_shape(scenario_and_dimensions):
    """Test the plot of the solution of the L-shape topology optimization.

    Here we consider the design value.
    """
    scenario_and_dimensions[0].post_process(
        "TopologyView",
        n_x=scenario_and_dimensions[1],
        n_y=scenario_and_dimensions[2],
        save=False,
        iterations=1,
    )


@image_comparison(["l_shape_solution_xphys"])
def test_l_shape_xphys(scenario_and_dimensions):
    """Test the plot of the solution of the L-shape topology optimization.

    Here we consider the value of an observable.
    """
    scenario_and_dimensions[0].post_process(
        "TopologyView",
        n_x=scenario_and_dimensions[1],
        n_y=scenario_and_dimensions[2],
        save=False,
        iterations=1,
        observable="xPhys",
    )


@image_comparison(["l_shape_solution_last_iter"])
def test_l_shape_last_iter(scenario_and_dimensions):
    """Test the plot of the solution of the L-shape topology optimization.

    Here we consider the last iteration.
    """
    scenario_and_dimensions[0].post_process(
        "TopologyView",
        n_x=scenario_and_dimensions[1],
        n_y=scenario_and_dimensions[2],
        save=False,
    )
