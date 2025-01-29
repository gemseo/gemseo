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

import re
from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest
from numpy import array
from numpy import ones
from numpy.testing import assert_almost_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.mdo.opt_as_mdo_scenario import OptAsMDOScenario
from gemseo.problems.mdo.scalable.parametric.scalable_problem import ScalableProblem


@pytest.fixture(scope="module")
def discipline() -> AnalyticDiscipline:
    """The discipline in the original optimization problem."""
    return AnalyticDiscipline(
        {"f": "100*(z_2-z_1**2)**2+(1-z_1)**2+100*(z_1-z_0**2)**2+(1-z_0)**2"},
        name="Rosenbrock",
    )


@pytest.mark.parametrize("initial_point", [None, array([-0.25, 0.75, -0.9])])
def test_basic(discipline, initial_point):
    """Check the scenario OptAsMDOScenario."""
    design_space = DesignSpace()
    design_space.add_variable("z_0", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_1", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_2", lower_bound=-1, upper_bound=1)
    if initial_point is not None:
        # This is the initial design
        # used in the example plot_opt_as_mdo of the documentation.
        design_space.set_current_value(initial_point)

    scenario = OptAsMDOScenario(discipline, "f", design_space, formulation_name="MDF")

    disciplines = scenario.disciplines
    assert disciplines[0].name == "Rosenbrock"
    assert disciplines[1].name == "L"
    assert disciplines[2].name == "D1"
    assert disciplines[3].name == "D2"

    scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
    assert_almost_equal(scenario.optimization_result.x_opt, ones(3))


@pytest.mark.parametrize("n_variables", range(3))
def test_less_than_3_design_variables(discipline, n_variables):
    """Check the error raised when the design space has less than 3 design variables."""
    design_space = DesignSpace()
    for i in range(n_variables):
        design_space.add_variable(f"z_{i}", lower_bound=-1, upper_bound=1)

    msg = "The design space must have at least three scalar design variables; got {}."
    with pytest.raises(ValueError, match=re.escape(msg.format(n_variables))):
        OptAsMDOScenario(discipline, "f", design_space)


def test_non_scalar_design_variables(discipline):
    """Check the error raised when the design space has less than 3 design variables."""
    design_space = DesignSpace()
    design_space.add_variable("z_1", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_2", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_3", size=2, lower_bound=-1, upper_bound=1)

    msg = "The design space must include scalar variables only."
    with pytest.raises(ValueError, match=re.escape(msg)):
        OptAsMDOScenario(discipline, "f", design_space)


def test_non_differentiable_link_discipline(discipline):
    """Check the error raised when the link discipline is not differentiable."""
    design_space = DesignSpace()
    design_space.add_variable("z_0", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_1", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_2", lower_bound=-1, upper_bound=1)
    with patch(
        "gemseo.problems.mdo.scalable.parametric.scalable_problem.ScalableProblem.differentiate_y",
        new_callable=PropertyMock,
        return_value=None,
    ):
        scenario = OptAsMDOScenario(
            discipline, "f", design_space, formulation_name="MDF"
        )

    with pytest.raises(
        ValueError, match=re.escape("The discipline L was not linearized.")
    ):
        scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)


def test_coupling_equations(discipline):
    """Check the use of coupling_equations and link_discipline argument."""
    design_space = DesignSpace()
    design_space.add_variable("z_0", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_1", lower_bound=-1, upper_bound=1)
    design_space.add_variable("z_2", lower_bound=-1, upper_bound=1)

    scalable_problem = ScalableProblem()
    coupling_equations = (
        scalable_problem.scalable_disciplines,
        scalable_problem.compute_y,
        scalable_problem.differentiate_y,
    )
    scenario = OptAsMDOScenario(
        discipline,
        "f",
        design_space,
        formulation_name="MDF",
        coupling_equations=coupling_equations,
    )

    disciplines = scenario.disciplines
    assert disciplines[0].name == "Rosenbrock"
    assert disciplines[1].name == "L"
    assert disciplines[2].name == "ScalableDiscipline[1]"
    assert disciplines[3].name == "ScalableDiscipline[2]"

    scenario.execute(algo_name="NLOPT_SLSQP", max_iter=100)
    assert_almost_equal(scenario.optimization_result.x_opt, ones(3))
