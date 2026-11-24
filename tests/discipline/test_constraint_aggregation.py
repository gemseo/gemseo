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
from numpy import ones_like
from numpy import vstack
from numpy.testing import assert_equal

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.discipline.constraint_aggregation import ConstraintAggregation
from gemseo.optimization.aggregation._functions import compute_upper_bound_ks_agg
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.optimization.power_2 import Power2
from gemseo.util.derivative.check.discipline import DisciplineJacobianChecker


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
    design_space.add_real_variable(
        "x", 3, lower_bound=-10.0, upper_bound=10.0, value=2.0
    )
    scenario = create_scenario(
        disciplines, "obj_f", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.add_constraint("constr", constraint_type=scenario.ConstraintType.INEQ)

    scenario.execute(SLSQP_Settings(max_iter=50))
    ref_sol = scenario.formulation.problem.solution

    disc_agg = create_discipline(
        "ConstraintAggregation",
        constraint_names=["constr"],
        aggregation_function="lower_bound_KS",
    )
    disc_agg.io.input_grammar.defaults = {"constr": array([1.0, 2.0])}
    assert DisciplineJacobianChecker(disc_agg).check(
        input_value={"constr": array([1.0, 2.0])}
    )

    disciplines = [disc_constr, disc_agg, obj_disc]
    design_space = create_design_space()
    design_space.add_real_variable(
        "x", 3, lower_bound=-10.0, upper_bound=10.0, value=2.0
    )
    scenario_agg = create_scenario(
        disciplines, "obj_f", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario_agg.add_constraint(
        "lower_bound_KS_constr", constraint_type=scenario_agg.ConstraintType.INEQ
    )

    scenario_agg.execute(SLSQP_Settings(max_iter=50))
    sol2 = scenario_agg.formulation.problem.solution

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
    checker = DisciplineJacobianChecker(disc_agg)
    assert checker.check(atol=1e-6, rtol=1e-6, step=1e-8)


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
    checker = DisciplineJacobianChecker(disc_agg)
    assert checker.check(atol=1e-6, rtol=1e-6, step=1e-8)


@pytest.mark.parametrize(
    "aggregation_attribute_value",
    [
        (ConstraintAggregation.AggregationFunction.LOWER_BOUND_KS, "lower_bound_KS"),
        (ConstraintAggregation.AggregationFunction.UPPER_BOUND_KS, "upper_bound_KS"),
    ],
)
def test_evaluation_function_as_enum(aggregation_attribute_value) -> None:
    """Check the use of AggregationFunction."""
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


@pytest.fixture
def namespaced_discipline() -> ConstraintAggregation:
    """A ConstraintAggregation whose first constraint has a namespace."""
    discipline = ConstraintAggregation(["c1", "c2"], "upper_bound_KS")
    discipline.add_namespace_to_input("c1", "ns")
    discipline.io.input_grammar.defaults = {
        "ns:c1": array([1.0, 2.0]),
        "c2": array([5.0]),
    }
    return discipline


def test_namespaces(namespaced_discipline) -> None:
    """Check that each constraint is aggregated into its own output."""
    namespaced_discipline.execute()
    output_data = namespaced_discipline.io.output_data
    assert_equal(
        output_data["upper_bound_KS_c1"],
        compute_upper_bound_ks_agg(array([1.0, 2.0])),
    )
    assert_equal(
        output_data["upper_bound_KS_c2"], compute_upper_bound_ks_agg(array([5.0]))
    )


def test_namespaces_jacobian(namespaced_discipline) -> None:
    """Check the Jacobian when an input has a namespace."""
    checker = DisciplineJacobianChecker(namespaced_discipline)
    assert checker.check(atol=1e-6, rtol=1e-6, step=1e-8)


@pytest.mark.parametrize(
    ("options", "c", "expected"),
    [
        ({}, array([1.0, 3.0, 2.0]), array([[0.0, 1.0, 0.0]])),
        (
            {"indices": [1, 0], "scale": array([2.0, 3.0])},
            array([0.0, 10.0]),
            array([[0.0, 2.0]]),
        ),
    ],
    ids=["default", "indices_and_array_scale"],
)
def test_max_jacobian(options, c, expected) -> None:
    """Check the Jacobian of the maximum."""
    discipline = ConstraintAggregation(["c"], "MAX", **options)
    jacobian = discipline.linearize({"c": c}, compute_all_jacobians=True)
    assert_equal(jacobian["MAX_c"]["c"], expected)


def test_max_with_indices_and_scale() -> None:
    """Check the maximum with indices and an array scale."""
    discipline = ConstraintAggregation(
        ["c"], "MAX", indices=[1, 0], scale=array([2.0, 3.0])
    )
    output_data = discipline.execute({"c": array([0.0, 10.0])})
    assert_equal(output_data["MAX_c"], array([20.0]))


def test_indices_of_each_constraint() -> None:
    """Check that the indices refer to the components of each constraint."""
    discipline = ConstraintAggregation(["c1", "c2"], "MAX", indices=[1])
    output_data = discipline.execute({
        "c1": array([1.0, 2.0]),
        "c2": array([4.0, 3.0]),
    })
    assert_equal(output_data["MAX_c1"], array([2.0]))
    assert_equal(output_data["MAX_c2"], array([3.0]))
