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

from __future__ import division, unicode_literals

from typing import Sequence

import pytest
from numpy import array
from numpy.linalg import norm

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdo_scenario import MDOScenario, MDOScenarioAdapter
from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
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
    grammar_type=MDOScenario.JSON_GRAMMAR_TYPE,  # type: str
):  # type: (...) -> MDOScenario
    """Build the scenario for SSBJ.

    Args:
        formulation: The name of the scenario formulation.
        grammar_type: The grammar type.

    Returns:
        The MDOScenario.
    """
    if grammar_type == MDOScenario.JSON_GRAMMAR_TYPE:
        disciplines = [
            SobieskiPropulsion(),
            SobieskiAerodynamics(),
            SobieskiMission(),
            SobieskiStructure(),
        ]
    elif grammar_type == MDOScenario.SIMPLE_GRAMMAR_TYPE:
        disciplines = [
            SobieskiPropulsionSG(),
            SobieskiAerodynamicsSG(),
            SobieskiMissionSG(),
            SobieskiStructureSG(),
        ]

    design_space = SobieskiProblem().read_design_space()
    scenario = MDOScenario(
        disciplines,
        formulation=formulation,
        objective_name="y_4",
        design_space=design_space,
        grammar_type=grammar_type,
        maximize_objective=True,
    )
    return scenario


@pytest.fixture()
def mdf_scenario():
    """Return a MDOScenario with MDF formulation and JSONGrammar.

    Returns:
        The MDOScenario.
    """
    return build_mdo_scenario("MDF")


@pytest.fixture()
def mdf_variable_grammar_scenario(request):
    """Return a MDOScenario with MDF formulation and custom grammar.

    Args:
        request: An auxiliary variable to retrieve the grammar type with
            pytest.mark.parametrize and the option `indirect=True`.

    Returns:
        The MDOScenario.
    """
    return build_mdo_scenario("MDF", request.param)


@pytest.fixture()
def idf_scenario():
    """Return a MDOScenario with IDF formulation and JSONGrammar.

    Return:
        The MDOScenario.
    """
    return build_mdo_scenario("IDF")


def test_scenario_state(mdf_scenario):
    stats = mdf_scenario.get_disciplines_statuses()

    assert len(stats) == len(mdf_scenario.disciplines)

    for disc in mdf_scenario.disciplines:
        assert disc.name in stats
        assert stats[disc.name] == "PENDING"


def test_add_user_defined_constraint_error(mdf_scenario):
    # Set the design constraints
    with pytest.raises(
        ValueError,
        match="Constraint type must be either 'eq' or 'ineq'; got 'foo' instead.",
    ):
        mdf_scenario.add_constraint(["g_1", "g_2", "g_3"], constraint_type="foo")

    mdf_scenario.set_differentiation_method(None)

    assert (
        mdf_scenario.formulation.opt_problem.differentiation_method == "no_derivatives"
    )


def test_save_optimization_history_exception(mdf_scenario):
    with pytest.raises(
        ValueError, match="Cannot export optimization history to file format: foo."
    ):
        mdf_scenario.save_optimization_history("file_path", file_format="foo")


@pytest.mark.parametrize(
    "file_format", [OptimizationProblem.GGOBI_FORMAT, OptimizationProblem.HDF5_FORMAT]
)
def test_save_optimization_history_format(mdf_scenario, file_format, tmp_wd):
    file_path = tmp_wd / "file_name"
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 2})
    mdf_scenario.save_optimization_history(str(file_path), file_format=file_format)
    assert file_path.exists()


def test_init_mdf(mdf_scenario):
    assert (
        sorted(["y_12", "y_21", "y_23", "y_31", "y_32"])
        == mdf_scenario.formulation.mda.strong_couplings
    )


def test_basic_idf(tmp_wd, idf_scenario):
    """"""
    posts = idf_scenario.posts

    assert len(posts) > 0

    for post in ["OptHistoryView", "Correlations", "QuadApprox"]:
        assert post in posts

    # Monitor in the console
    idf_scenario.xdsmize(
        outdir=str(tmp_wd), json_output=True, html_output=True, open_browser=False
    )

    assert (tmp_wd / "xdsm.json").exists()
    assert (tmp_wd / "xdsm.html").exists()


def test_backup_error(tmp_wd, mdf_scenario):
    """"""
    expected_message = (
        "Conflicting options for history backup, "
        "cannot pre load optimization history and erase it!"
    )
    with pytest.raises(ValueError, match=expected_message):
        mdf_scenario.set_optimization_history_backup(
            __file__, erase=True, pre_load=True
        )

    with pytest.raises(IOError):
        mdf_scenario.set_optimization_history_backup(
            __file__, erase=False, pre_load=True
        )


def test_backup_0(tmp_wd, mdf_scenario):
    """Test the optimization backup with generation of plots during convergence.

    tests that when used, the backup does not call the original objective
    """
    filename = "opt_history.h5"
    mdf_scenario.set_optimization_history_backup(
        filename, erase=True, pre_load=False, generate_opt_plot=True
    )
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 2})
    assert len(mdf_scenario.formulation.opt_problem.database) == 2

    assert (tmp_wd / filename).exists()

    opt_read = OptimizationProblem.import_hdf(filename)

    assert len(opt_read.database) == len(mdf_scenario.formulation.opt_problem.database)

    mdf_scenario.set_optimization_history_backup(filename, erase=True, pre_load=False)
    assert not (tmp_wd / filename).exists()


@pytest.mark.parametrize(
    "mdf_variable_grammar_scenario",
    [MDOScenario.SIMPLE_GRAMMAR_TYPE, MDOScenario.JSON_GRAMMAR_TYPE],
    indirect=True,
)
def test_backup_1(tmp_wd, mdf_variable_grammar_scenario):
    """Test the optimization backup with generation of plots during convergence.

    tests that when used, the backup does not call the original objective
    """
    filename = "opt_history.h5"
    mdf_variable_grammar_scenario.set_optimization_history_backup(
        filename, erase=False, pre_load=True, generate_opt_plot=False
    )
    mdf_variable_grammar_scenario.execute({"algo": "SLSQP", "max_iter": 2})
    opt_read = OptimizationProblem.import_hdf(filename)

    assert len(opt_read.database) == len(
        mdf_variable_grammar_scenario.formulation.opt_problem.database
    )

    assert (
        norm(
            array(
                mdf_variable_grammar_scenario.formulation.opt_problem.database.get_x_history()
            )
            - array(
                mdf_variable_grammar_scenario.formulation.opt_problem.database.get_x_history()
            )
        )
        == 0.0
    )


def test_typeerror_formulation():
    disciplines = [SobieskiPropulsion()]
    design_space = SobieskiProblem().read_design_space()

    expected_message = (
        "Formulation must be specified by its name; "
        "please use GEMSEO_PATH to specify custom formulations."
    )
    with pytest.raises(TypeError, match=expected_message):
        MDOScenario(disciplines, 1, "y_4", design_space)


@pytest.mark.parametrize(
    "mdf_variable_grammar_scenario",
    [MDOScenario.SIMPLE_GRAMMAR_TYPE, MDOScenario.JSON_GRAMMAR_TYPE],
    indirect=True,
)
def test_get_optimization_results(mdf_variable_grammar_scenario):
    """Test the optimization results accessor.

    Test the case when the Optimization results are available.
    """
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

    mdf_variable_grammar_scenario.optimization_result = opt_results

    optimum = mdf_variable_grammar_scenario.get_optimum()

    assert optimum.x_opt == x_opt
    assert optimum.f_opt == f_opt
    assert optimum.constraints_values == constraints_values
    assert optimum.constraints_grad == constraints_grad
    assert optimum.is_feasible == is_feasible


def test_get_optimization_results_empty(mdf_scenario):
    """Test the optimization results accessor.

    Test the case when the Optimization results are not available (e.g. when the execute
    method has not been executed).
    """
    assert mdf_scenario.get_optimum() is None


def test_adapter(tmp_wd, idf_scenario):
    """Test the adapter."""
    # Monitor in the console
    idf_scenario.xdsmize(
        True,
        print_statuses=True,
        outdir=str(tmp_wd),
        json_output=True,
        html_output=True,
    )

    idf_scenario.default_inputs = {
        "max_iter": 1,
        "algo": "SLSQP",
        idf_scenario.ALGO_OPTIONS: {"max_iter": 1},
    }

    inputs = ["x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(idf_scenario, inputs, outputs)
    gen = MDOFunctionGenerator(adapter)
    func = gen.get_function(inputs, outputs)
    x_shared = array([0.06000319728113519, 60000, 1.4, 2.5, 70, 1500])
    f_x1 = func(x_shared)
    f_x2 = func(x_shared)

    assert f_x1 == f_x2
    assert len(idf_scenario.formulation.opt_problem.database) == 1

    x_shared = array([0.09, 60000, 1.4, 2.5, 70, 1500])
    func(x_shared)


def test_adapter_error(idf_scenario):
    """Test the adapter."""
    inputs = ["x_shared"]
    outputs = ["y_4"]

    with pytest.raises(
        ValueError, match="Can't compute inputs from scenarios: missing_input."
    ):
        MDOScenarioAdapter(idf_scenario, inputs + ["missing_input"], outputs)

    with pytest.raises(
        ValueError, match="Can't compute outputs from scenarios: missing_output."
    ):
        MDOScenarioAdapter(idf_scenario, inputs, outputs + ["missing_output"])


def test_repr_str(idf_scenario):
    assert str(idf_scenario) == idf_scenario.name

    expected = [
        "MDOScenario",
        "   Disciplines: "
        "SobieskiPropulsion SobieskiAerodynamics SobieskiMission SobieskiStructure",
        "   MDOFormulation: IDF",
        "   Algorithm: None",
    ]
    assert repr(idf_scenario) == "\n".join(expected)


def test_xdsm_filename(tmp_path, idf_scenario):
    """Tests the export path dir for xdsm."""
    outfilename = "my_xdsm.html"
    idf_scenario.xdsmize(
        outdir=tmp_path, outfilename=outfilename, latex_output=False, html_output=True
    )
    assert (tmp_path / outfilename).is_file()


@pytest.mark.parametrize("observables", [["y_12"], ["y_23"]])
def test_add_observable(
    mdf_scenario,  # type: MDOScenario
    observables,  # type: Sequence[str]
):
    """Test adding observables from discipline outputs.

    Args:
         mdf_scenario: A fixture for the MDOScenario.
         observables: A list of observables.
    """
    mdf_scenario.add_observable(observables)
    new_observables = mdf_scenario.formulation.opt_problem.observables
    for new_observable, expected_observable in zip(new_observables, observables):
        assert new_observable.name == expected_observable


def test_add_observable_not_available(
    mdf_scenario,  # type: MDOScenario
):
    """Test adding an observable which is not available in any discipline.

    Args:
         mdf_scenario: A fixture for the MDOScenario.
    """
    msg = "^No discipline known by formulation MDF has all outputs named .*"
    with pytest.raises(ValueError, match=msg):
        mdf_scenario.add_observable("toto")


def test_database_name(mdf_scenario):
    """Check the name of the database."""
    assert mdf_scenario.formulation.opt_problem.database.name == "MDOScenario"
