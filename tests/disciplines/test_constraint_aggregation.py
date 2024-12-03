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
from numpy import allclose
from numpy import array
from numpy import concatenate
from numpy import ones
from numpy import ones_like
from numpy import vstack
from numpy.testing import assert_equal

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.disciplines.constraint_aggregation import ConstraintAggregation
from gemseo.problems.optimization.power_2 import Power2


@pytest.fixture
def disc_constr():
    """A Sellar problem discipline."""
    problem = Power2()
    constraints = problem.constraints[:2]

    def cstr(x):
        constr = concatenate([cstr.evaluate(x) for cstr in constraints])
        return constr  # noqa: RET504

    def jac(x):
        return vstack([cstr.jac(x) for cstr in constraints])

    return create_discipline(
        "AutoPyDiscipline", py_func=cstr, py_jac=jac, use_arrays=True
    )


def obj(x):
    """Dummy sum objective function."""
    obj_f = array([sum(x)])
    return obj_f  # noqa: RET504


def test_aggregation_discipline(disc_constr) -> None:
    """Tests the constraint aggregation discipline in a scenario, with analytic
    derivatives and adjoint."""
    obj_disc = create_discipline(
        "AutoPyDiscipline", py_func=obj, py_jac=lambda x: ones_like(x), use_arrays=True
    )
    disciplines = [disc_constr, obj_disc]
    design_space = create_design_space()
    design_space.add_variable(
        "x", 3, lower_bound=-10, upper_bound=10, value=2 * ones(3)
    )
    scenario = create_scenario(
        disciplines, "obj_f", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.add_constraint("constr", constraint_type="ineq")

    scenario.execute(algo_name="SLSQP", max_iter=50)
    ref_sol = scenario.formulation.optimization_problem.solution

    disc_agg = create_discipline(
        "ConstraintAggregation",
        constraint_names=["constr"],
        aggregation_function="lower_bound_KS",
    )
    disc_agg.io.input_grammar.defaults = {"constr": array([1.0, 2.0])}
    assert disc_agg.check_jacobian(input_data={"constr": array([1.0, 2.0])})

    disciplines = [disc_constr, disc_agg, obj_disc]
    design_space = create_design_space()
    design_space.add_variable(
        "x", 3, lower_bound=-10, upper_bound=10, value=2 * ones(3)
    )
    scenario_agg = create_scenario(
        disciplines, "obj_f", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario_agg.add_constraint("lower_bound_KS_constr", constraint_type="ineq")

    scenario_agg.execute(algo_name="SLSQP", max_iter=50)
    sol2 = scenario_agg.formulation.optimization_problem.solution

    assert allclose(sol2.x_opt, ref_sol.x_opt, rtol=1e-2)


@pytest.mark.parametrize("indices", [None, array([0]), array([1])])
@pytest.mark.parametrize(
    "aggregation_function",
    ["lower_bound_KS", "upper_bound_KS", "IKS", "POS_SUM", "SUM"],
)
@pytest.mark.parametrize("input_val", [(1.0, 2.0), (0.0, 0.0), (-1.0, -2.0)])
def test_constr_jac(disc_constr, aggregation_function, indices, input_val) -> None:
    """Checks the Jacobian of the AggregationDiscipline."""
    disc_agg = create_discipline(
        "ConstraintAggregation",
        constraint_names=["constr"],
        aggregation_function=aggregation_function,
        indices=indices,
    )
    disc_agg.io.input_grammar.defaults = {"constr": array(input_val)}
    assert disc_agg.check_jacobian(threshold=1e-6, step=1e-8)


@pytest.mark.parametrize("scale", [1.0, array([2.0, 3.0])])
@pytest.mark.parametrize(
    "aggregation_function",
    ["lower_bound_KS", "upper_bound_KS", "IKS", "POS_SUM", "SUM"],
)
@pytest.mark.parametrize("input_val", [(1.0, 2.0), (0.0, 0.0), (-1.0, -2.0)])
def test_constr_jac_scale(disc_constr, aggregation_function, scale, input_val) -> None:
    """Checks the Jacobian of the AggregationDiscipline with scale effect."""
    disc_agg = create_discipline(
        "ConstraintAggregation",
        constraint_names=["constr"],
        aggregation_function=aggregation_function,
        scale=scale,
    )
    disc_agg.io.input_grammar.defaults = {"constr": array(input_val)}
    assert disc_agg.check_jacobian(threshold=1e-6, step=1e-8)


@pytest.mark.parametrize(
    "aggregation_attribute_value",
    [
        (ConstraintAggregation.EvaluationFunction.LOWER_BOUND_KS, "lower_bound_KS"),
        (ConstraintAggregation.EvaluationFunction.UPPER_BOUND_KS, "upper_bound_KS"),
    ],
)
def test_evaluation_function_as_enum(aggregation_attribute_value) -> None:
    """Check the use of EvaluationFunction."""
    discipline = create_discipline(
        "ConstraintAggregation",
        constraint_names=["constr"],
        aggregation_function=aggregation_attribute_value[0],
    )
    discipline.io.input_grammar.defaults = {"constr": array([1.0, 2.0])}
    discipline.execute()
    output_data_with_enum = discipline.io.get_output_data()

    discipline = create_discipline(
        "ConstraintAggregation",
        constraint_names=["constr"],
        aggregation_function=aggregation_attribute_value[1],
    )
    discipline.io.input_grammar.defaults = {"constr": array([1.0, 2.0])}
    discipline.execute()
    output_data_without_enum = discipline.io.get_output_data()
    assert_equal(output_data_without_enum, output_data_with_enum)
