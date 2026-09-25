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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo import create_scenario
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.formulation.disciplinary_opt import DisciplinaryOpt
from gemseo.formulation.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.scipy_local.settings.cobyla import COBYLA_Settings
from gemseo.space.design import DesignSpace
from tests.core.function.test_mdo_discipline_adapter import (
    DisciplineWithNonNumericInput,
)


@pytest.mark.parametrize(
    ("options", "expected_jac"),
    [
        ({}, array([2.0])),
        ({"differentiated_input_names_substitute": ["a"]}, array([2.0])),
        ({"differentiated_input_names_substitute": ["b"]}, array([3.0])),
        ({"differentiated_input_names_substitute": ["a", "b"]}, array([2.0, 3.0])),
        ({"differentiated_input_names_substitute": ["b", "a"]}, array([3.0, 2.0])),
    ],
)
def test_jac_wrt_dv_or_non_dv(options, expected_jac):
    """Check the Jacobian wrt design or non-design input variables."""
    discipline = AnalyticDiscipline({"f": "2*a+3*b", "c": "2*a+3*b", "o": "2*a+3*b"})

    design_space = DesignSpace()
    design_space.add_variable("a")

    problem = OptimizationProblem(design_space)

    formulation = DisciplinaryOpt(
        problem, [discipline], DisciplinaryOpt_Settings(**options)
    )
    problem.objective = formulation.create_objective(["f"])
    constraint = formulation.create_constraint(["c"])
    problem.add_constraint(constraint)
    formulation.add_observable(["o"])

    for function in [problem.objective, problem.constraints[0], problem.observables[0]]:
        assert_equal(function.evaluate(array([1])), array([2.0]))
        assert_equal(function.jac(array([1])), expected_jac)


def test_scenario_with_non_numeric_discipline_input() -> None:
    """Check that a non-numeric, non-design discipline input does not break a scenario.

    Regression test: a discipline input that is not a number or an array
    (e.g. a `list`) and that is not a design variable used to make the
    `DisciplineAdapter` crash while computing sizes for all the discipline
    inputs, when it only needs the sizes of the design variables.
    """
    discipline = DisciplineWithNonNumericInput()

    design_space = DesignSpace()
    design_space.add_variable("z", value=0.5, lower_bound=-100.0, upper_bound=50.0)

    scenario = create_scenario(
        [discipline],
        "f",
        design_space,
        formulation_settings_model=DisciplinaryOpt_Settings(),
    )
    scenario.execute(COBYLA_Settings(max_iter=3))

    # The sizes of the discipline inputs that are neither design variables nor
    # differentiated inputs must not have leaked into the formulation.
    assert scenario.formulation.variable_sizes == {
        name: variable.size for name, variable in design_space.variables.items()
    }
