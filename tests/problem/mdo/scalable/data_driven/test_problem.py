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
# INITIAL AUTHORS - initial API and implementation and/or
#                   initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import os
from pathlib import Path

import pytest

from gemseo.formulation import BiLevel_Settings
from gemseo.optimization import NLOPT_COBYLA_Settings
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.mdo.scalable.data_driven.problem import ScalableProblem
from gemseo.problem.mdo.sobieski.discipline import SobieskiAerodynamics
from gemseo.problem.mdo.sobieski.discipline import SobieskiMission
from gemseo.problem.mdo.sobieski.discipline import SobieskiPropulsion
from gemseo.problem.mdo.sobieski.discipline import SobieskiStructure
from gemseo.util.pickle import from_pickle
from tests.marks import requires_numpy_2

pytestmark = requires_numpy_2

n_samples = 10


@pytest.fixture(scope="module")
def scalable_problem():
    design_variables = ["x_shared", "x_1", "x_2", "x_3"]
    objective_function = "y_4"
    ineq_constraints = ["g_1", "g_2"]
    eq_constraints = ["g_3"]
    aero = SobieskiAerodynamics()
    propu = SobieskiPropulsion()
    struct = SobieskiStructure()
    mission = SobieskiMission()
    disciplines = [aero, propu, struct, mission]
    disc_names = [disc.name for disc in disciplines]
    datasets = []
    for name in disc_names:
        dataset = from_pickle(Path(__file__).parent / f"{name}.pkl")
        datasets.append(dataset)
    return ScalableProblem(
        datasets, design_variables, objective_function, eq_constraints, ineq_constraints
    )


def test_print(scalable_problem) -> None:
    assert "Sizes" in str(scalable_problem)


def test_plot_n2_chart(scalable_problem, tmp_wd) -> None:
    """"""
    scalable_problem.plot_n2_chart()
    assert os.path.exists("n2.pdf")


def test_plot_coupling_graph(scalable_problem, tmp_wd) -> None:
    """"""
    scalable_problem.plot_coupling_graph()
    assert os.path.exists("coupling_graph.pdf")


def test_plot_1d_interpolations(scalable_problem, tmp_wd) -> None:
    """"""
    files = scalable_problem.plot_1d_interpolations(directory=str(tmp_wd))
    assert len(files) > 0
    for fname in files:
        assert os.path.exists(fname)


def test_plot_dependencies(scalable_problem, tmp_wd) -> None:
    """"""
    files = scalable_problem.plot_dependencies(directory=str(tmp_wd))
    assert len(files) > 0
    for fname in files:
        assert os.path.exists(fname)


def test_create_scenario(scalable_problem) -> None:
    """"""
    scalable_problem.create_scenario()


@pytest.mark.parametrize(
    ("sub_optimizer_settings", "expected_settings"),
    [(None, SLSQP_Settings()), (NLOPT_COBYLA_Settings(), NLOPT_COBYLA_Settings())],
)
def test_create_bilevel_scenario(
    scalable_problem, sub_optimizer_settings, expected_settings
) -> None:
    """Check the creation of a scenario using a bi-level formulation."""
    scenario = scalable_problem.create_scenario(
        formulation_settings=BiLevel_Settings(),
        sub_optimizer_settings=sub_optimizer_settings,
    )
    assert list(scenario.design_space) == ["x_shared"]
    adapters = scenario.formulation.scenario_adapters
    assert [set(adapter.scenario.design_space) for adapter in adapters] == [
        {"x_2"},
        {"x_3"},
        {"x_1"},
    ]
    output_names = [name for adapter in adapters for name in adapter.io.output_grammar]
    assert len(output_names) == len(set(output_names))
    for adapter in adapters:
        assert adapter.scenario._algorithm_settings == expected_settings


def test_execute_bilevel_scenario(scalable_problem) -> None:
    """Check the execution of a scenario using a bi-level formulation."""
    scenario = scalable_problem.create_scenario(
        formulation_settings=BiLevel_Settings(),
        sub_optimizer_settings=SLSQP_Settings(max_iter=2),
    )
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=2))
    assert scenario.formulation.problem.database
    for adapter in scenario.formulation.scenario_adapters:
        assert adapter.scenario.formulation.problem.database


def test_statistics(scalable_problem, enable_discipline_statistics) -> None:
    """"""
    scalable_problem.create_scenario()
    scalable_problem.get_execution_duration()
    scalable_problem.n_calls  # noqa: B018
    scalable_problem.n_calls_linearize  # noqa: B018
    scalable_problem.scenario.execute(SLSQP_Settings(max_iter=100))
    scalable_problem.status  # noqa: B018
