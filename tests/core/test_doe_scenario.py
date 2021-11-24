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
#      :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import unicode_literals

import pytest
from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.api import create_discipline, create_scenario
from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)
from gemseo.problems.sobieski.wrappers_sg import (
    SobieskiAerodynamicsSG,
    SobieskiMissionSG,
    SobieskiPropulsionSG,
    SobieskiStructureSG,
)


def build_mdo_scenario(
    formulation,  # type: str
    grammar_type=DOEScenario.JSON_GRAMMAR_TYPE,  # type: str
):  # type: (...) -> DOEScenario
    """Build the DOE scenario for SSBJ.

    Args:
        formulation: The name of the DOE scenario formulation.
        grammar_type: The grammar type.

    Returns:
        The DOE scenario.
    """
    if grammar_type == DOEScenario.JSON_GRAMMAR_TYPE:
        disciplines = [
            SobieskiPropulsion(),
            SobieskiAerodynamics(),
            SobieskiMission(),
            SobieskiStructure(),
        ]
    elif grammar_type == DOEScenario.SIMPLE_GRAMMAR_TYPE:
        disciplines = [
            SobieskiPropulsionSG(),
            SobieskiAerodynamicsSG(),
            SobieskiMissionSG(),
            SobieskiStructureSG(),
        ]

    design_space = SobieskiProblem().read_design_space()
    scenario = DOEScenario(
        disciplines,
        formulation=formulation,
        objective_name="y_4",
        design_space=design_space,
        grammar_type=grammar_type,
        maximize_objective=True,
    )
    return scenario


@pytest.fixture()
def mdf_variable_grammar_doe_scenario(request):
    """Return a DOEScenario with MDF formulation and custom grammar.

    Args:
        request: An auxiliary variable to retrieve the grammar type with
            pytest.mark.parametrize and the option `indirect=True`.
    """
    return build_mdo_scenario("MDF", request.param)


@pytest.mark.usefixtures("tmp_wd")
@pytest.mark.skip_under_windows
def test_parallel_doe_hdf_cache(caplog):
    disciplines = create_discipline(
        [
            "SobieskiStructure",
            "SobieskiPropulsion",
            "SobieskiAerodynamics",
            "SobieskiMission",
        ]
    )
    path = "cache.h5"
    for disc in disciplines:
        disc.set_cache_policy(disc.HDF5_CACHE, cache_hdf_file=path)

    scenario = create_scenario(
        disciplines,
        "DisciplinaryOpt",
        "y_4",
        SobieskiProblem().read_design_space(),
        maximize_objective=True,
        scenario_type="DOE",
    )

    n_samples = 10
    input_data = {
        "n_samples": n_samples,
        "algo": "lhs",
        "algo_options": {"n_processes": 2},
    }
    scenario.execute(input_data)
    scenario.print_execution_metrics()
    assert len(scenario.formulation.opt_problem.database) == n_samples
    for disc in disciplines:
        assert disc.cache.get_length() == n_samples

    input_data = {
        "n_samples": n_samples,
        "algo": "lhs",
        "algo_options": {"n_processes": 2, "n_samples": n_samples},
    }
    scenario.execute(input_data)
    expected_log = "Double definition of algorithm option n_samples, keeping value: {}."
    assert expected_log.format(n_samples) in caplog.text


@pytest.mark.parametrize(
    "mdf_variable_grammar_doe_scenario",
    [DOEScenario.SIMPLE_GRAMMAR_TYPE, DOEScenario.JSON_GRAMMAR_TYPE],
    indirect=True,
)
def test_doe_scenario(mdf_variable_grammar_doe_scenario):
    """Test the execution of a DOEScenario with different grammars.

    Args:
        mdf_variable_grammar_doe_scenario: The DOEScenario.
    """

    n_samples = 10
    input_data = {
        "n_samples": n_samples,
        "algo": "lhs",
        "algo_options": {"n_processes": 1},
    }
    mdf_variable_grammar_doe_scenario.execute(input_data)
    mdf_variable_grammar_doe_scenario.print_execution_metrics()
    assert (
        len(mdf_variable_grammar_doe_scenario.formulation.opt_problem.database)
        == n_samples
    )


def test_warning_when_missing_option(caplog):
    """Check that a warning is correctly logged when an option is unknown."""
    discipline = AnalyticDiscipline(name="func", expressions_dict={"y": "2*x"})
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0)
    scenario = DOEScenario([discipline], "DisciplinaryOpt", "y", design_space)
    scenario.execute(
        {
            "algo": "CustomDOE",
            "algo_options": {"samples": array([[1.0]]), "unknown_option": 1},
        }
    )
    expected_log = "Driver CustomDOE has no option {}, option is ignored."
    assert expected_log.format("n_samples") not in caplog.text
    assert expected_log.format("unknown_option") in caplog.text
