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
"""Tests for the welded beam problem."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo.enum import ApproximationMode
from gemseo.enum import ConstraintType
from gemseo.formulation import DisciplinaryOpt_Settings
from gemseo.optimization import SLSQP_Settings
from gemseo.problem.optimization._welded_beam import Welding
from gemseo.problem.optimization._welded_beam.data import catalog
from gemseo.problem.optimization._welded_beam.data import get_properties
from gemseo.problem.optimization._welded_beam.design_space import create_design_space
from gemseo.problem.optimization._welded_beam.discipline import WeldedBeam
from gemseo.scenario import MDOScenario
from gemseo.util.derivative.check.discipline import DisciplineJacobianChecker
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.space import DesignSpace
    from gemseo.util.typing import RealArray

constraint_names = (
    "shear_stress",
    "bending_stress",
    "thickness_difference",
    "buckling",
    "deflection",
)

dimension_names = ("weld_thickness", "weld_length", "bar_height", "bar_thickness")

# The optimum of each welding, reached with steel, as (h, l, t, b) and f.
optima = {
    Welding.TWO_SIDED: ((0.25, 4.40085668, 8.25, 0.25), 2.129332),
    Welding.FOUR_SIDED: ((0.1875, 1.78210225, 8.25, 0.25), 1.955301),
}


def create_input_data(
    h: float, length: float, t: float, b: float, material: str = "steel"
) -> dict[str, RealArray]:
    """Create input data of the welded beam.

    Args:
        h: The weld thickness.
        length: The weld length.
        t: The bar height.
        b: The bar thickness.
        material: The label of the material.

    Returns:
        The input data.
    """
    return {
        name: array([value])
        for name, value in zip(dimension_names, (h, length, t, b), strict=True)
    } | get_properties(material)


def test_design_space():
    """Check the design space."""
    design_space = create_design_space()
    variables = design_space.variables
    assert {name: variable.type for name, variable in variables.items()} == {
        "material": "catalog",
        "weld_thickness": "discrete",
        "weld_length": "real",
        "bar_height": "discrete",
        "bar_thickness": "discrete",
    }
    assert variables["material"].catalog is catalog
    for name in ("weld_thickness", "bar_thickness"):
        assert_equal(variables[name].choices, 0.0625 * array(range(1, 33)))
    assert_equal(variables["bar_height"].choices, 0.0625 * array(range(32, 321)))
    assert_equal(variables["weld_length"].lower_bound, (0.1,))
    assert_equal(variables["weld_length"].upper_bound, (10.0,))
    current_value = design_space.get_current_value(as_dict=True)
    expected_value = {
        "material": array([0]),
        "weld_thickness": array([0.5]),
        "weld_length": array([5.0]),
        "bar_height": array([10.0]),
        "bar_thickness": array([0.5]),
    }
    assert current_value.keys() == expected_value.keys()
    for name, value in expected_value.items():
        assert_equal(current_value[name], value)


@pytest.mark.parametrize(
    ("welding", "dimensions", "expected"),
    [
        pytest.param(
            Welding.TWO_SIDED,
            optima[Welding.TWO_SIDED][0],
            (optima[Welding.TWO_SIDED][1], 0.0, -0.012672, 0.0, -0.067008, -0.937449),
            id="two-sided",
        ),
        pytest.param(
            Welding.FOUR_SIDED,
            optima[Welding.FOUR_SIDED][0],
            (
                optima[Welding.FOUR_SIDED][1],
                0.0,
                -0.012672,
                -0.0625,
                -0.067008,
                -0.937449,
            ),
            id="four-sided",
        ),
        pytest.param(
            Welding.FOUR_SIDED,
            (0.3, 9.0, 3.0, 1.2),
            (5.175756, -0.490301, 0.555556, -0.9, -50.202181, -0.728988),
            id="four-sided-l>t",
        ),
    ],
)
def test_execute(welding, dimensions, expected):
    """Check the outputs at the optimum of each welding and with a long weld."""
    output_data = WeldedBeam(welding).execute(create_input_data(*dimensions))
    assert_allclose(
        [output_data[name][0] for name in ("cost", *constraint_names)],
        expected,
        atol=1e-6,
    )


def test_invalid_welding(snapshot):
    """Check that an invalid welding raises an error."""
    with assert_exception(ValueError, snapshot):
        WeldedBeam("four-sided")


@pytest.mark.parametrize("welding", tuple(Welding))
@pytest.mark.parametrize(
    "input_data",
    [
        pytest.param({}, id="l<t"),
        pytest.param(create_input_data(0.3, 3.0, 3.0, 1.2, "cast_iron"), id="l=t"),
        pytest.param(create_input_data(0.3, 9.0, 3.0, 1.2, "cast_iron"), id="l>t"),
    ],
)
def test_jacobian(welding, input_data):
    """Check the analytic Jacobian, with a weld length below, at and above the height.

    At the tie, the complex step does not change the real part of the radii,
    so it differentiates the expressions for a weld length smaller than the height,
    as the analytic Jacobian does.
    """
    assert DisciplineJacobianChecker(WeldedBeam(welding)).check(
        input_data, approximation_mode=ApproximationMode.COMPLEX_STEP, step=1e-30
    )


@pytest.mark.parametrize("welding", tuple(Welding))
def test_pickle(welding):
    """Check that a pickled welded beam computes the same outputs and Jacobian."""
    discipline = WeldedBeam(welding)
    unpickled_discipline = pickle.loads(pickle.dumps(discipline))
    input_data = create_input_data(0.3, 9.0, 3.0, 1.2, "cast_iron")
    for disc in (discipline, unpickled_discipline):
        disc.execute(input_data)
        disc.linearize(input_data, compute_all_jacobians=True)

    assert_equal(
        dict(unpickled_discipline.io.output_data), dict(discipline.io.output_data)
    )
    assert_equal(unpickled_discipline.jac, discipline.jac)


def create_scenario(discipline: WeldedBeam, design_space: DesignSpace) -> MDOScenario:
    """Create a scenario minimizing the cost of the welded beam.

    Args:
        discipline: The welded beam.
        design_space: The design space.

    Returns:
        The scenario.
    """
    scenario = MDOScenario(
        [discipline], design_space, formulation_settings=DisciplinaryOpt_Settings()
    )
    scenario.add_objective("cost")
    for name in constraint_names:
        scenario.add_constraint(name, ConstraintType.INEQ)

    return scenario


@pytest.mark.parametrize("welding", tuple(Welding))
def test_optimum(welding):
    """Check the optimum of each welding.

    Only the weld length is optimized.
    The material is steel
    and the other dimensions are set to their optimal values.
    """
    (h, length, t, b), cost = optima[welding]
    discipline = WeldedBeam(welding)
    discipline.io.input_grammar.defaults.update(create_input_data(h, length, t, b))
    design_space = create_design_space().filter("weld_length")
    scenario = create_scenario(discipline, design_space)
    scenario.execute(SLSQP_Settings(max_iter=100))
    result = scenario.optimization_result
    assert result.f_opt == pytest.approx(cost, rel=1e-6)
    assert_allclose(result.x_opt, (length,), rtol=1e-6)


def test_relaxation():
    """Check that the driver can relax the discrete dimensions.

    The material is steel and the four dimensions are optimized
    as continuous variables.
    The result is feasible, no worse than the discrete optimum
    and equal to the continuous optimum.
    """
    design_space = create_design_space().filter(dimension_names)
    scenario = create_scenario(WeldedBeam(), design_space)
    scenario.execute(SLSQP_Settings(max_iter=200, relax_discrete_variables=True))
    result = scenario.optimization_result
    assert result.is_feasible
    assert result.f_opt <= optima[Welding.TWO_SIDED][1]
    assert result.f_opt == pytest.approx(2.101324, rel=1e-5)
