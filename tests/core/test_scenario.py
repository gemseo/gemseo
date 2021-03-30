# -*- coding: utf-8 -*-
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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import os
from builtins import open
from os.path import exists, join

import pytest
from future import standard_library
from numpy import array

from gemseo import SOFTWARE_NAME
from gemseo.algos.opt_result import OptimizationResult
from gemseo.api import configure_logger
from gemseo.core.function import MDOFunctionGenerator
from gemseo.core.mdo_scenario import MDOScenario, MDOScenarioAdapter
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)
from gemseo.third_party.junitxmlreq import link_to

standard_library.install_aliases()


LOGGER = configure_logger(SOFTWARE_NAME)


def create_design_space():
    return SobieskiProblem().read_design_space()


def build_mdo_scenario(formulation="MDF"):
    """
    Build the scenario for SSBJ

    :param formulation: Default value = 'MDF'

    """
    disciplines = [
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
        SobieskiStructure(),
    ]
    design_space = create_design_space()
    scenario = MDOScenario(
        disciplines,
        formulation=formulation,
        objective_name="y_4",
        design_space=design_space,
        maximize_objective=True,
    )
    return scenario


def test_add_user_defined_constraint_error():
    scenario = build_mdo_scenario("MDF")
    stats = scenario.get_disciplines_statuses()
    assert len(stats) == len(scenario.disciplines)
    for disc in scenario.disciplines:
        assert disc.name in stats
        assert stats[disc.name] == "PENDING"
    # Set the design constraints
    with pytest.raises(Exception):
        scenario.add_constraint(["g_1", "g_2", "g_3"], "None")
    with pytest.raises(Exception):
        scenario.save_optimization_history("file_path", file_format="toto")

    scenario.set_differentiation_method(None)
    assert scenario.formulation.opt_problem.differentiation_method == "no_derivatives"


@link_to(
    "Req-INT-1",
    "Req-INT-2",
    "Req-INT-3",
    "Req-INT-3.1",
    "Req-INT-3.2",
    "Req-INT-3.3",
    "Req-MDO-4.3",
    "Req-MDO-11",
    "Req-SC-1.1",
    "Req-SC-2",
    "Req-SC-2.1",
    "Req-SC-3",
    "Req-SC-4",
    "Req-SC-5",
    "Req-SC-6",
)
def test_init_mdf():
    """ """
    scs = [build_mdo_scenario("MDF")]
    for scenario in scs:
        assert len(scenario.formulation.mda.strong_couplings) == 8
        for coupling in scenario.formulation.mda.strong_couplings:
            assert coupling.startswith("y_")


def test_basic_idf(tmpdir):
    """ """
    scenario = build_mdo_scenario("IDF")
    posts = scenario.posts

    assert len(posts) > 0
    for post in ["OptHistoryView", "Correlations", "QuadApprox"]:
        assert post in posts

        # Monitor in the console
    scenario.xdsmize(
        outdir=str(tmpdir), json_output=True, html_output=True, open_browser=False
    )

    assert exists(join(str(tmpdir), "xdsm.json"))
    assert exists(join(str(tmpdir), "xdsm.html"))


def test_backup():
    sc = build_mdo_scenario()
    with pytest.raises(ValueError):
        sc.set_optimization_history_backup(__file__, erase=True, pre_load=True)
    with pytest.raises(IOError):
        sc.set_optimization_history_backup(__file__, erase=False, pre_load=True)

    filename = "temporary_file.txt"
    if exists(filename):
        os.remove(filename)
    with open(filename, "w") as f:
        f.write("something")
    sc.set_optimization_history_backup(
        filename, erase=True, pre_load=False, generate_opt_plot=True
    )
    sc.execute({"algo": "SLSQP", "max_iter": 10})


def test_typeerror_formulation():
    disciplines = [
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
        SobieskiStructure(),
    ]
    design_space = create_design_space()

    with pytest.raises(TypeError):
        MDOScenario(disciplines, 1, "y_4", design_space)


def test_get_optimization_results():
    """Test the optimization results accessor

    Test the case when the Optimization results are available
    """

    scenario = build_mdo_scenario()

    x_opt = [1.0, 2.0]
    f_opt = 3
    constraints_values = [4.0, 5.0]
    constraints_grad = [6.0, 7.0]
    is_feasible = True

    opt_results = OptimizationResult(
        x_opt=x_opt,
        f_opt=f_opt,
        constraints_values=constraints_values,
        constraints_grad=constraints_grad,
        is_feasible=is_feasible,
    )

    scenario.optimization_result = opt_results

    optimum = scenario.get_optimum()

    assert optimum.x_opt == x_opt
    assert optimum.f_opt == f_opt
    assert optimum.constraints_values == constraints_values
    assert optimum.constraints_grad == constraints_grad
    assert optimum.is_feasible == is_feasible


def test_get_optimization_results_empty():
    """Test the optimization results accessor

    Test the case when the Optimization results are not available (e.g.
    when the execute method has not been executed)
    """

    scenario = build_mdo_scenario()
    assert scenario.get_optimum() is None


def test_adapter(tmpdir):
    """Test the adapter """

    disciplines = [
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
        SobieskiStructure(),
    ]

    design_space = create_design_space()
    scenario = MDOScenario(
        disciplines,
        formulation="IDF",
        objective_name="y_4",
        design_space=design_space,
        maximize_objective=True,
    )
    # Monitor in the console
    scenario.xdsmize(
        True,
        print_statuses=True,
        outdir=str(tmpdir),
        json_output=True,
        html_output=True,
    )

    scenario.default_inputs = {
        "max_iter": 1,
        "algo": "SLSQP",
        scenario.ALGO_OPTIONS: {"max_iter": 1},
    }

    inputs = ["x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs)
    gen = MDOFunctionGenerator(adapter)
    func = gen.get_function(inputs, outputs)
    x_shared = array([0.06000319728113519, 60000, 1.4, 2.5, 70, 1500])
    f_x1 = func(x_shared)
    f_x2 = func(x_shared)
    assert f_x1 == f_x2
    assert len(scenario.formulation.opt_problem.database) == 1
    x_shared = array([0.09, 60000, 1.4, 2.5, 70, 1500])

    func(x_shared)

    with pytest.raises(ValueError):
        MDOScenarioAdapter(scenario, inputs + ["missing_input"], outputs)

    with pytest.raises(ValueError):
        MDOScenarioAdapter(scenario, inputs, outputs + ["missing_output"])
