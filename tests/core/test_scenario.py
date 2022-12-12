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
from __future__ import annotations

import pickle
import re
import unittest
from pathlib import Path
from typing import Sequence

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.scenario_adapter import MDOScenarioAdapter
from gemseo.problems.sobieski._disciplines_sg import SobieskiAerodynamicsSG
from gemseo.problems.sobieski._disciplines_sg import SobieskiMissionSG
from gemseo.problems.sobieski._disciplines_sg import SobieskiPropulsionSG
from gemseo.problems.sobieski._disciplines_sg import SobieskiStructureSG
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from numpy import array
from numpy import complex128
from numpy import float64
from numpy import int64
from numpy.linalg import norm
from numpy.testing import assert_equal


def build_mdo_scenario(
    formulation: str,
    grammar_type: str = MDOScenario.JSON_GRAMMAR_TYPE,
) -> MDOScenario:
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

    design_space = SobieskiProblem().design_space
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
    file_path = Path("file_name")
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
    idf_scenario.xdsmize(json_output=True)
    assert Path("xdsm.json").exists()
    assert Path("xdsm.html").exists()


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
        mdf_scenario.set_optimization_history_backup(__file__, pre_load=True)


@pytest.mark.parametrize("each_iter", [False, True])
def test_backup_0(tmp_wd, mdf_scenario, each_iter):
    """Test the optimization backup with generation of plots during convergence.

    Test that, when used, the backup does not call the original objective.
    """
    file_path = Path("opt_history.h5")
    mdf_scenario.set_optimization_history_backup(
        file_path, erase=True, generate_opt_plot=True, each_new_iter=each_iter
    )
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 2})
    assert len(mdf_scenario.formulation.opt_problem.database) == 2

    assert file_path.exists()

    opt_read = OptimizationProblem.import_hdf(file_path)

    assert len(opt_read.database) == len(mdf_scenario.formulation.opt_problem.database)

    mdf_scenario.set_optimization_history_backup(file_path, erase=True)
    assert not file_path.exists()


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
        filename, pre_load=True
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
    design_space = SobieskiProblem().design_space

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
    x_opt = array([1.0, 2.0])
    f_opt = array([3.0])
    constraints_values = {"g": array([4.0, 5.0])}
    constraints_grad = {"g": array([6.0, 7.0])}
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

    assert_equal(optimum.x_opt, x_opt)
    assert_equal(optimum.f_opt, f_opt)
    assert_equal(optimum.constraints_values, constraints_values)
    assert_equal(optimum.constraints_grad, constraints_grad)
    assert optimum.is_feasible is is_feasible


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
        True, print_statuses=True, outdir=str(tmp_wd), json_output=True
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
        "SobieskiAerodynamics SobieskiMission SobieskiPropulsion SobieskiStructure",
        "   MDO formulation: IDF",
    ]
    assert repr(idf_scenario) == "\n".join(expected)


def test_xdsm_filename(tmp_wd, idf_scenario):
    """Tests the export path dir for xdsm."""
    outfilename = "my_xdsm.html"
    idf_scenario.xdsmize(outfilename=outfilename)
    assert Path(outfilename).is_file()


@pytest.mark.parametrize("observables", [["y_12"], ["y_23"]])
def test_add_observable(
    mdf_scenario: MDOScenario,
    observables: Sequence[str],
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
    mdf_scenario: MDOScenario,
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


@unittest.mock.patch("timeit.default_timer", new=lambda: 1)
def test_run_log(mdf_scenario, caplog):
    """Check the log message of Scenario._run."""
    mdf_scenario._run_algorithm = lambda: None
    mdf_scenario.name = "ABC Scenario"
    mdf_scenario._run()
    strings = [
        "*** Start ABC Scenario execution ***",
        "*** End ABC Scenario execution (time: 0:00:00) ***",
    ]
    for string in strings:
        assert string in caplog.text


def test_clear_history_before_run(mdf_scenario):
    """Check that clear_history_before_run is correctly used in Scenario._run."""
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 1})
    assert len(mdf_scenario.formulation.opt_problem.database) == 1

    def run_algorithm_mock():
        pass

    mdf_scenario._run_algorithm = run_algorithm_mock
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 1})
    assert len(mdf_scenario.formulation.opt_problem.database) == 1

    mdf_scenario.clear_history_before_run = True
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 1})
    assert len(mdf_scenario.formulation.opt_problem.database) == 0


@pytest.mark.parametrize(
    "activate,text",
    [
        (True, "Scenario Execution Statistics"),
        (False, "The discipline counters are disabled."),
    ],
)
def test_print_execution_metrics(mdf_scenario, caplog, activate, text):
    """Check the print of the execution metrics w.r.t.

    activate_counters.
    """
    activate_counters = MDODiscipline.activate_counters
    MDODiscipline.activate_counters = activate
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 1})
    mdf_scenario.print_execution_metrics()
    assert text in caplog.text
    MDODiscipline.activate_counters = activate_counters


def test_get_execution_metrics(mdf_scenario):
    """Check the string returned byecution_metrics."""
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 1})
    expected = re.compile(
        "Scenario Execution Statistics\n"
        "   Discipline: SobieskiPropulsion\n"
        "      Executions number: 10\n"
        "      Execution time: .* s\n"
        "      Linearizations number: 1\n"
        "   Discipline: SobieskiAerodynamics\n"
        "      Executions number: 10\n"
        "      Execution time: .* s\n"
        "      Linearizations number: 1\n"
        "   Discipline: SobieskiMission\n"
        "      Executions number: 1\n"
        "      Execution time: .* s\n"
        "      Linearizations number: 1\n"
        "   Discipline: SobieskiStructure\n"
        "      Executions number: 10\n"
        "      Execution time: .* s\n"
        "      Linearizations number: 1\n"
        "   Total number of executions calls: 31\n"
        "   Total number of linearizations: 4"
    )
    assert expected.match(str(mdf_scenario._Scenario__get_execution_metrics()))


def mocked_export_to_dataset(
    name: str | None = None,
    by_group: bool = True,
    categorize: bool = True,
    opt_naming: bool = True,
    export_gradients: bool = False,
) -> Dataset:
    """A mock for OptimizationProblem.export_to_dataset."""
    return (
        name,
        by_group,
        categorize,
        opt_naming,
        export_gradients,
    )


def test_export_to_dataset(mdf_scenario):
    """Check that export_to_dataset calls OptimizationProblem.export_to_dataset."""
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 1})
    mdf_scenario.export_to_dataset = mocked_export_to_dataset
    dataset = mdf_scenario.export_to_dataset(
        name=1, by_group=2, categorize=3, opt_naming=4, export_gradients=5
    )
    assert dataset == (1, 2, 3, 4, 5)


@pytest.fixture
def complex_step_scenario() -> MDOScenario:
    """The scenario to be used by test_complex_step."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)

    class MyDiscipline(MDODiscipline):
        """The identity discipline f computing y = f(x) = x."""

        def __init__(self) -> None:
            super().__init__()
            self.input_grammar.update(["x"])
            self.output_grammar.update(["y"])

        def _run(self) -> None:
            self.local_data["y"] = self.local_data["x"]

    scenario = MDOScenario([MyDiscipline()], "DisciplinaryOpt", "y", design_space)
    scenario.set_differentiation_method(scenario.COMPLEX_STEP)
    return scenario


@pytest.mark.parametrize("normalize_design_space", [False, True])
def test_complex_step(complex_step_scenario, normalize_design_space):
    """Check that complex step approximation works correctly."""
    complex_step_scenario.execute(
        {
            "algo": "SLSQP",
            "max_iter": 10,
            "algo_options": {"normalize_design_space": normalize_design_space},
        }
    )

    assert complex_step_scenario.optimization_result.x_opt[0] == 0.0


@pytest.fixture
def sinus_use_case() -> tuple[AnalyticDiscipline, DesignSpace]:
    """The sinus discipline and its design space."""
    discipline = AnalyticDiscipline({"y": "sin(2*pi*x)"})
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)
    return discipline, design_space


@pytest.mark.parametrize(
    "maximize,standardize,expr,val",
    [
        (False, False, "minimize y(x)", -1.0),
        (False, True, "minimize y(x)", -1.0),
        (True, False, "maximize y(x)", 1.0),
        (True, True, "minimize -y(x)", -1.0),
    ],
)
def test_use_standardized_objective(
    sinus_use_case, maximize, standardize, expr, val, caplog
):
    """Check that the setter use_standardized_objective works correctly."""
    discipline, design_space = sinus_use_case
    scenario = MDOScenario(
        [discipline],
        formulation="MDF",
        objective_name="y",
        maximize_objective=maximize,
        design_space=design_space,
    )
    assert scenario.use_standardized_objective
    scenario.use_standardized_objective = standardize
    assert scenario.use_standardized_objective is standardize
    scenario.execute({"algo": "SLSQP", "max_iter": 10})
    assert expr in caplog.text
    assert f"Objective: {val}" in caplog.text
    assert f"obj={int(val)}" in caplog.text


@pytest.mark.parametrize(
    "cast_default_inputs_to_complex, expected_dtype",
    [(True, complex128), (False, float64)],
)
def test_complex_casting(
    cast_default_inputs_to_complex, expected_dtype, mdf_scenario: MDOScenario
):
    """Check the automatic casting of default inputs when complex_step is selected.

    Args:
        cast_default_inputs_to_complex: Whether to cast the default inputs of the
            scenario's disciplines to complex.
        expected_dtype: The expected ``dtype`` after setting the differentiation method.
        mdf_scenario: A fixture for the MDOScenario.
    """
    for discipline in mdf_scenario.disciplines:
        for value in discipline.default_inputs.values():
            assert value.dtype == float64

    mdf_scenario.set_differentiation_method(
        mdf_scenario.COMPLEX_STEP,
        cast_default_inputs_to_complex=cast_default_inputs_to_complex,
    )
    for discipline in mdf_scenario.disciplines:
        for value in discipline.default_inputs.values():
            assert value.dtype == expected_dtype


@pytest.fixture
def scenario_with_non_float_variables() -> MDOScenario:
    """Create an ``MDOScenario`` from an ``AnalyticDiscipline`` with non-float inputs.

    Returns:
        The MDOScenario.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)

    discipline = AnalyticDiscipline({"y": "x"})
    discipline.input_grammar.update(["z"])
    discipline.input_grammar.update(["w"])
    discipline.default_inputs["z"] = "some_str"
    discipline.default_inputs["w"] = array(1, dtype=int64)

    return MDOScenario([discipline], "DisciplinaryOpt", "y", design_space)


@pytest.mark.parametrize(
    "cast_default_inputs_to_complex, expected_dtype",
    [(True, complex128), (False, float64)],
)
def test_complex_casting_with_non_float_variables(
    cast_default_inputs_to_complex, expected_dtype, scenario_with_non_float_variables
):
    """Test that the scenario will not cast non-float variables to complex.

    Args:
        cast_default_inputs_to_complex: Whether to cast the float default inputs of the
            scenario's disciplines to complex.
        expected_dtype: The expected ``dtype`` after setting the differentiation method.
        scenario_with_non_float_variables: Fixture that returns an ``MDOScenario`` with
            an AnalyticDiscipline that has integer and string inputs.
    """
    scenario_with_non_float_variables.set_differentiation_method(
        scenario_with_non_float_variables.COMPLEX_STEP,
        cast_default_inputs_to_complex=cast_default_inputs_to_complex,
    )

    assert (
        scenario_with_non_float_variables.formulation.design_space._current_value[
            "x"
        ].dtype
        == complex128
    )
    assert (
        scenario_with_non_float_variables.disciplines[0].default_inputs["x"].dtype
        == expected_dtype
    )
    assert isinstance(
        scenario_with_non_float_variables.disciplines[0].default_inputs["z"], str
    )
    assert (
        scenario_with_non_float_variables.disciplines[0].default_inputs["w"].dtype
        == int64
    )


def test_check_disciplines():
    """Test that an exception is raised when two disciplines compute the same output."""
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=1.0, value=0.5)

    discipline_1 = AnalyticDiscipline({"y": "x"}, name="foo")
    discipline_2 = AnalyticDiscipline({"y": "x + 1"}, name="bar")
    with pytest.raises(
        ValueError,
        match="Two disciplines, among "
        f"which {discipline_2.name}, "
        "compute the same output: {'y'}",
    ):
        MDOScenario([discipline_1, discipline_2], "DisciplinaryOpt", "y", design_space)


def test_lib_serialization(tmp_wd, mdf_scenario):
    """Test the serialization of an MDOScenario with an instantiated opt_lib.

    Args:
        mdf_scenario: A fixture for the MDOScenario.
    """
    mdf_scenario.execute({"algo": "SLSQP", "max_iter": 1})
    mdf_scenario.formulation.opt_problem.reset(database=False, design_space=False)

    with open("scenario.pkl", "wb") as file:
        pickle.dump(mdf_scenario, file)

    with open("scenario.pkl", "rb") as file:
        pickled_scenario = pickle.load(file)

    assert pickled_scenario._lib is None

    pickled_scenario.execute({"algo": "SLSQP", "max_iter": 1})

    assert pickled_scenario._lib.internal_algo_name == "SLSQP"
