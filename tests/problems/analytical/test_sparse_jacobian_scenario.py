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

from typing import TYPE_CHECKING

import pytest

from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.core.chain import MDOChain
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.disciplines.linear_combination import LinearCombination
from gemseo.disciplines.splitter import Splitter

if TYPE_CHECKING:
    from gemseo.core.scenario import Scenario


@pytest.fixture(scope="module", params=["DisciplinaryOpt", "IDF"])
def scenario(request) -> Scenario:
    """Optimization scenario involving MDOFunctions with sparse Jacobians."""
    design_space = DesignSpace()
    design_space.add_variable("alpha", size=2, l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("beta", size=2, l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("gamma", size=2, l_b=0.0, u_b=1.0, value=0.5)

    disciplines = [
        LinearCombination(
            input_names=["alpha", "beta", "gamma"],
            output_name="delta",
            input_coefficients={"alpha": 1.0, "beta": -2.0, "gamma": 3.0},
            offset=-2.0,
            input_size=2,
        ),
        LinearCombination(
            input_names=["alpha", "beta", "gamma"],
            output_name="eta",
            input_coefficients={"alpha": 1.0, "beta": +1.0, "gamma": 1.0},
            offset=-4.0,
            input_size=2,
        ),
        Splitter(
            input_name="delta",
            output_names_to_input_indices={"delta_1": 0, "delta_2": 1},
        ),
        LinearCombination(
            input_names=["delta_1", "delta_2"],
            output_name="rho",
            input_coefficients={"delta_1": 1.0, "delta_2": +1.0},
            offset=0,
        ),
    ]

    discipline = MDOChain(disciplines)
    discipline.set_linear_relationships()

    scenario = create_scenario(
        disciplines=[discipline],
        formulation=request.param,
        design_space=design_space,
        objective_name="rho",
    )

    scenario.add_constraint("eta", MDOFunction.ConstraintType.INEQ)

    return scenario


def test_problem_is_linear(scenario):
    """Tests that optimization problems are linear."""
    assert (
        scenario.formulation.opt_problem.pb_type
        == scenario.formulation.opt_problem.ProblemType.LINEAR
    )


def test_execution(scenario):
    """Tests the execution of scenario with sparse Jacobians."""
    scenario.execute({"algo": "LINEAR_INTERIOR_POINT", "max_iter": 1000})
    out = scenario.formulation.opt_problem.solution
    assert pytest.approx(out.f_opt) == -8.0
