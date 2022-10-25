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
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.problems.analytical.power_2 import Power2
from numpy import allclose
from numpy import array
from numpy import concatenate
from numpy import ones
from numpy import ones_like
from numpy import vstack


@pytest.fixture
def disc_constr():
    """A Sellar problem discipline."""
    problem = Power2()
    constraints = problem.constraints[:2]

    def cstr(x):
        constr = concatenate([cstr(x) for cstr in constraints])
        return constr

    def jac(x):
        return vstack([cstr.jac(x) for cstr in constraints])

    return create_discipline(
        "AutoPyDiscipline", py_func=cstr, py_jac=jac, use_arrays=True
    )


def obj(x):
    """Dummy sum objective function."""
    obj_f = array([sum(x)])
    return obj_f


def test_aggregation_discipline(disc_constr):
    """Tests the constraint aggregation discipline in a scenario, with analytic
    derivatives and adjoint."""
    obj_disc = create_discipline(
        "AutoPyDiscipline", py_func=obj, py_jac=lambda x: ones_like(x), use_arrays=True
    )
    disciplines = [disc_constr, obj_disc]
    design_space = create_design_space()
    design_space.add_variable("x", 3, l_b=-10, u_b=10, value=2 * ones(3))
    scenario = create_scenario(disciplines, "DisciplinaryOpt", "obj_f", design_space)
    scenario.add_constraint("constr", "ineq")

    scenario.execute({"algo": "SLSQP", "max_iter": 50})
    ref_sol = scenario.formulation.opt_problem.solution

    disc_agg = create_discipline(
        "ConstrAggegationDisc", constr_data_names=["constr"], method_name="KS"
    )
    disc_agg.default_inputs = {"constr": array([1.0, 2.0])}
    assert disc_agg.check_jacobian(input_data={"constr": array([1.0, 2.0])})

    disciplines = [disc_constr, disc_agg, obj_disc]
    design_space = create_design_space()
    design_space.add_variable("x", 3, l_b=-10, u_b=10, value=2 * ones(3))
    scenario_agg = create_scenario(
        disciplines, "DisciplinaryOpt", "obj_f", design_space
    )
    scenario_agg.add_constraint("KS_constr", "ineq")

    scenario_agg.execute({"algo": "SLSQP", "max_iter": 50})
    sol2 = scenario_agg.formulation.opt_problem.solution

    assert allclose(sol2.x_opt, ref_sol.x_opt, rtol=1e-2)


def test_wrong_meth():
    """Tests the constraint aggregation discipline in a scenario, with analytic
    derivatives and adjoint."""
    with pytest.raises(ValueError, match="Unsupported aggregation method named"):
        create_discipline(
            "ConstrAggegationDisc", constr_data_names=["constr"], method_name="unknown"
        )


@pytest.mark.parametrize("indices", (None, array([0]), array([1])))
@pytest.mark.parametrize("method_name", ["KS", "IKS"])
@pytest.mark.parametrize("input_val", [(1.0, 2.0), (0.0, 0.0), (-1.0, -2.0)])
def test_constr_jac(disc_constr, method_name, indices, input_val):
    """Checks the Jacobian of the AggregationDiscipline."""
    disc_agg = create_discipline(
        "ConstrAggegationDisc",
        constr_data_names=["constr"],
        method_name=method_name,
        indices=indices,
    )
    disc_agg.default_inputs = {"constr": array(input_val)}
    assert disc_agg.check_jacobian(threshold=1e-6, step=1e-8)


@pytest.mark.parametrize("scale", (1.0, array([2.0, 3.0])))
@pytest.mark.parametrize("method_name", ["KS", "IKS"])
@pytest.mark.parametrize("input_val", [(1.0, 2.0), (0.0, 0.0), (-1.0, -2.0)])
def test_constr_jac_scale(disc_constr, method_name, scale, input_val):
    """Checks the Jacobian of the AggregationDiscipline with scale effect."""
    disc_agg = create_discipline(
        "ConstrAggegationDisc",
        constr_data_names=["constr"],
        method_name=method_name,
        scale=scale,
    )
    disc_agg.default_inputs = {"constr": array(input_val)}
    assert disc_agg.check_jacobian(threshold=1e-6, step=1e-8)
