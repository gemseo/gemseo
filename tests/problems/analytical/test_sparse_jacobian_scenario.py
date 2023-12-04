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
from gemseo.core.chain import MDOChain
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.disciplines.linear_combination import LinearCombination
from gemseo.disciplines.splitter import Splitter


@pytest.fixture()
def linear_sparse_scenario():
    discipline1 = LinearCombination(
        input_names=["alpha", "beta", "gamma"],
        output_name="delta",
        input_coefficients={"alpha": 1.0, "beta": -2.0, "gamma": 3.0},
        offset=-2.0,
    )
    discipline2 = LinearCombination(
        input_names=["alpha", "beta", "gamma"],
        output_name="eta",
        input_coefficients={"alpha": 1.0, "beta": +1.0, "gamma": 1.0},
        offset=-4.0,
    )
    discipline3 = Splitter(
        input_name="delta", output_names_to_input_indices={"delta_1": 0, "delta_2": 1}
    )
    discipline4 = LinearCombination(
        input_names=["delta_1", "delta_2"],
        output_name="rho",
        input_coefficients={"delta_1": 1.0, "delta_2": +1.0},
        offset=0,
    )
    design_space = DesignSpace()
    design_space.add_variable("alpha", size=2, l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("beta", size=2, l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("gamma", size=2, l_b=0.0, u_b=1.0, value=0.5)
    disc5 = MDOChain([discipline1, discipline2, discipline3, discipline4])
    disc5.set_linear_relationships()
    scenario = create_scenario(
        disciplines=[disc5],
        formulation="DisciplinaryOpt",
        design_space=design_space,
        objective_name="rho",
    )
    scenario.add_constraint("eta", constraint_type=MDOFunction.ConstraintType.INEQ)
    return scenario


@pytest.fixture()
def linear_idf_scenario():
    discipline1 = LinearCombination(
        input_names=["x1", "x2", "y2"],
        output_name="y1",
        input_coefficients={"x1": 1.0, "x2": 1.0, "y2": 1.0},
        offset=0.0,
    )
    discipline2 = LinearCombination(
        input_names=["y1"],
        output_name="y2",
        input_coefficients={"y1": -1.0},
        offset=0.0,
    )
    discipline3 = LinearCombination(
        input_names=["y1"],
        output_name="f",
        input_coefficients={"y1": 1.0},
        offset=0.0,
    )
    design_space = DesignSpace()
    design_space.add_variable("x1", size=1, l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("x2", size=1, l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("y1", size=1, l_b=0.0, u_b=1.0, value=0.5)
    design_space.add_variable("y2", size=1, l_b=0.0, u_b=1.0, value=0.5)
    disc5 = MDOChain([discipline1, discipline2, discipline3])
    disc5.set_linear_relationships()
    return create_scenario(
        disciplines=[disc5],
        formulation="IDF",
        design_space=design_space,
        objective_name="f",
    )


def test_linear_problem(linear_sparse_scenario):
    assert (
        linear_sparse_scenario.formulation.opt_problem.pb_type
        == linear_sparse_scenario.formulation.opt_problem.ProblemType.LINEAR
    )


def test_linear_problem_idf(linear_idf_scenario):
    assert (
        linear_idf_scenario.formulation.opt_problem.pb_type
        == linear_idf_scenario.formulation.opt_problem.ProblemType.LINEAR
    )


def test_sparse_discipline_jac(linear_sparse_scenario):
    assert MDOChain(linear_sparse_scenario.disciplines).check_jacobian(
        input_data=linear_sparse_scenario.formulation.design_space.get_current_value(
            as_dict=True
        )
    )


def test_execution(linear_sparse_scenario):
    linear_sparse_scenario.execute({"algo": "LINEAR_INTERIOR_POINT", "max_iter": 1000})
    out = linear_sparse_scenario.formulation.opt_problem.solution
    assert pytest.approx(out.f_opt) == -8.0


def test_execution_idf(linear_idf_scenario):
    linear_idf_scenario.execute({"algo": "LINEAR_INTERIOR_POINT", "max_iter": 1000})
    out = linear_idf_scenario.formulation.opt_problem.solution
    assert pytest.approx(out.f_opt, abs=1e-5) == 0.0
