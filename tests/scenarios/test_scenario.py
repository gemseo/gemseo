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
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from numpy import array
from numpy import complex128
from numpy import float64
from numpy import int64
from numpy.linalg import norm
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal

from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.core.discipline import Discipline
from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)
from gemseo.core.mdo_functions.function_from_discipline import FunctionFromDiscipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.mdo_linear_function import MDOLinearFunction
from gemseo.datasets.dataset import Dataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiAerodynamicsSG
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiMissionSG
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiPropulsionSG
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiStructureSG
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.scenarios.mdo_scenario import MDOScenario

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

PARENT_PATH = Path(__file__).parent
SOBIESKI_HDF5_PATH = PARENT_PATH / "mdf_backup.h5"


def build_mdo_scenario(
    formulation_name: str,
    grammar_type: Discipline.GrammarType = Discipline.GrammarType.JSON,
) -> MDOScenario:
    """Build the scenario for SSBJ.

    Args:
        formulation_name: The name of the scenario formulation.
        grammar_type: The grammar type.

    Returns:
        The MDOScenario.
    """
    if grammar_type == Discipline.GrammarType.JSON:
        disciplines = [
            SobieskiPropulsion(),
            SobieskiAerodynamics(),
            SobieskiMission(),
            SobieskiStructure(),
        ]
    elif grammar_type == Discipline.GrammarType.SIMPLE:
        disciplines = [
            SobieskiPropulsionSG(),
            SobieskiAerodynamicsSG(),
            SobieskiMissionSG(),
            SobieskiStructureSG(),
        ]

    design_space = SobieskiDesignSpace()
    scenario = MDOScenario(
        disciplines,
        "y_4",
        design_space,
        formulation_name=formulation_name,
        maximize_objective=True,
    )
    for c_name in ["g_1", "g_2", "g_3"]:
        scenario.add_constraint(c_name, constraint_type=MDOFunction.ConstraintType.INEQ)
    return scenario


@pytest.fixture
def mdf_scenario():
    """Return a MDOScenario with MDF formulation and JSONGrammar.

    Returns:
        The MDOScenario.
    """
    return build_mdo_scenario("MDF")


@pytest.fixture
def mdf_variable_grammar_scenario(request):
    """Return a MDOScenario with MDF formulation and custom grammar.

    Args:
        request: An auxiliary variable to retrieve the grammar type with
            pytest.mark.parametrize and the option `indirect=True`.

    Returns:
        The MDOScenario.
    """
    return build_mdo_scenario("MDF", request.param)


@pytest.fixture
def idf_scenario():
    """Return a MDOScenario with IDF formulation and JSONGrammar.

    Return:
        The MDOScenario.
    """
    return build_mdo_scenario("IDF")


def test_add_user_defined_constraint_error(mdf_scenario) -> None:
    # Set the design constraints
    mdf_scenario.set_differentiation_method(
        mdf_scenario.DifferentiationMethod.NO_DERIVATIVE
    )

    assert (
        mdf_scenario.formulation.optimization_problem.differentiation_method
        == mdf_scenario.DifferentiationMethod.NO_DERIVATIVE
    )


@pytest.mark.parametrize("file_format", OptimizationProblem.HistoryFileFormat)
def test_save_optimization_history_format(mdf_scenario, file_format, tmp_wd) -> None:
    file_path = Path("file_name")
    mdf_scenario.execute(algo_name="SLSQP", max_iter=2)
    mdf_scenario.save_optimization_history(str(file_path), file_format=file_format)
    assert file_path.exists()


def test_init_mdf(mdf_scenario) -> None:
    assert (
        sorted(["y_12", "y_21", "y_23", "y_31", "y_32"])
        == mdf_scenario.formulation.mda.coupling_structure.strong_couplings
    )


def test_basic_idf(tmp_wd, idf_scenario) -> None:
    """"""
    posts = idf_scenario.posts
    assert len(posts) > 0
    for post in ["OptHistoryView", "Correlations", "QuadApprox"]:
        assert post in posts

    # Monitor in the console
    idf_scenario.xdsmize(save_json=True)
    assert Path("xdsm.json").exists()
    assert Path("xdsm.html").exists()


def test_backup_error(tmp_wd, mdf_scenario) -> None:
    """"""
    expected_message = (
        "Conflicting options for history backup, "
        "cannot pre load optimization history and erase it!"
    )
    with pytest.raises(ValueError, match=expected_message):
        mdf_scenario.set_optimization_history_backup(__file__, erase=True, load=True)

    with pytest.raises(IOError):
        mdf_scenario.set_optimization_history_backup(__file__, load=True)


@pytest.mark.parametrize("load", [True, False])
def test_optimization_hist_backup_pre_load(tmp_wd, mdf_scenario, load) -> None:
    """Test the load option of the optimization history backup."""
    mdf_scenario.set_optimization_history_backup(SOBIESKI_HDF5_PATH, load=load)
    database = mdf_scenario.formulation.optimization_problem.database
    assert len(database) == 4 if load else len(database) == 0


@pytest.mark.parametrize("at_each_function_call", [True, False])
def test_optimization_hist_backup_each_store(
    tmp_wd, mdf_scenario, at_each_function_call
) -> None:
    """Test the backup execution at each iteration."""
    file_path = Path("opt_history.h5")
    mdf_scenario.set_optimization_history_backup(
        file_path,
        at_each_iteration=False,
        at_each_function_call=at_each_function_call,
    )

    inputs = array([
        5.0e-02,
        4.5e04,
        1.6e00,
        5.5e00,
        5.5e01,
        1.0e03,
        2.5e-01,
        1.0e00,
        1.0e00,
        5.0e-01,
    ])
    y_4 = {"-y_4": -535.7821319229388}
    g_1 = {
        "g_1": array([0.035, -0.00666667, -0.0275, -0.04, -0.04833333, -0.09, -0.15])
    }

    mdf_scenario.formulation.optimization_problem.database.store(inputs, y_4)
    mdf_scenario.formulation.optimization_problem.database.store(inputs, g_1)

    if at_each_function_call:
        backup_problem = OptimizationProblem.from_hdf(file_path)
        assert "-y_4" in backup_problem.database[inputs]
        assert "g_1" in backup_problem.database[inputs]
    else:
        assert not file_path.exists()


@pytest.mark.parametrize("erase", [False, True])
def test_optimization_hist_backup_erase(tmp_wd, mdf_scenario, erase) -> None:
    """Test that the erase option deletes the backup file as intended."""
    file_path = Path("opt_history.h5")
    with open(file_path, "w"):
        pass
    mdf_scenario.set_optimization_history_backup(file_path, erase=erase)
    file_exists = file_path.exists()
    assert not file_exists if erase else file_exists


@pytest.mark.parametrize("plot", [True, False])
def test_optimization_hist_backup_plot(tmp_wd, mdf_scenario, plot) -> None:
    """Test the plot creation with the plot option.

    Four iterations are needed to generate the Hessian approximation plot.
    """
    file_path = Path("opt_history.h5")
    mdf_scenario.set_optimization_history_backup(file_path, plot=plot)
    mdf_scenario.execute(algo_name="SLSQP", max_iter=4)
    for suffix in [
        "ineq_constraints",
        "objective",
        "variables",
        "x_xstar",
    ]:
        file_exists = Path(f"opt_history_{suffix}.png").exists()
        assert file_exists if plot else not file_exists


@pytest.mark.parametrize(
    "mdf_variable_grammar_scenario",
    [Discipline.GrammarType.SIMPLE, Discipline.GrammarType.JSON],
    indirect=True,
)
def test_backup_1(tmp_wd, mdf_variable_grammar_scenario) -> None:
    """Test the optimization backup with generation of plots during convergence.

    tests that when used, the backup does not call the original objective
    """
    filename = "opt_history.h5"
    mdf_variable_grammar_scenario.set_optimization_history_backup(filename, load=True)
    mdf_variable_grammar_scenario.execute(algo_name="SLSQP", max_iter=2)
    opt_read = OptimizationProblem.from_hdf(filename)

    assert len(opt_read.database) == len(
        mdf_variable_grammar_scenario.formulation.optimization_problem.database
    )

    assert (
        norm(
            array(
                mdf_variable_grammar_scenario.formulation.optimization_problem.database.get_x_vect_history()
            )
            - array(
                mdf_variable_grammar_scenario.formulation.optimization_problem.database.get_x_vect_history()
            )
        )
        == 0.0
    )


@pytest.mark.parametrize(
    "mdf_variable_grammar_scenario",
    [Discipline.GrammarType.SIMPLE, Discipline.GrammarType.JSON],
    indirect=True,
)
def test_get_optimization_results(mdf_variable_grammar_scenario) -> None:
    """Test the optimization results accessor.

    Test the case when the Optimization results are available.
    """
    x_opt = array([1.0, 2.0])
    f_opt = array([3.0])
    constraint_values = {"g": array([4.0, 5.0])}
    constraints_grad = {"g": array([6.0, 7.0])}
    is_feasible = True

    opt_results = OptimizationResult(
        x_opt=x_opt,
        f_opt=f_opt,
        constraint_values=constraint_values,
        constraints_grad=constraints_grad,
        is_feasible=is_feasible,
    )

    mdf_variable_grammar_scenario.optimization_result = opt_results
    optimum = mdf_variable_grammar_scenario.optimization_result

    assert_equal(optimum.x_opt, x_opt)
    assert_equal(optimum.f_opt, f_opt)
    assert_equal(optimum.constraint_values, constraint_values)
    assert_equal(optimum.constraints_grad, constraints_grad)
    assert optimum.is_feasible is is_feasible


def test_get_optimization_results_empty(mdf_scenario) -> None:
    """Test the optimization results accessor.

    Test the case when the Optimization results are not available (e.g. when the execute
    method has not been executed).
    """
    assert mdf_scenario.optimization_result is None


def test_adapter(tmp_wd, idf_scenario) -> None:
    """Test the adapter."""
    # Monitor in the console
    idf_scenario.xdsmize(True, log_workflow_status=True, save_json=True)

    idf_scenario.set_algorithm(algo_name="SLSQP", max_iter=1)

    inputs = ["x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(idf_scenario, inputs, outputs)
    gen = DisciplineAdapterGenerator(adapter)
    func = gen.get_function(inputs, outputs)
    x_shared = array([0.06000319728113519, 60000, 1.4, 2.5, 70, 1500])
    f_x1 = func.evaluate(x_shared)
    f_x2 = func.evaluate(x_shared)

    assert f_x1 == f_x2
    assert len(idf_scenario.formulation.optimization_problem.database) == 1

    x_shared = array([0.09, 60000, 1.4, 2.5, 70, 1500])
    func.evaluate(x_shared)


def test_adapter_error(idf_scenario) -> None:
    """Test the adapter."""
    inputs = ["x_shared"]
    outputs = ["y_4"]

    with pytest.raises(
        ValueError,
        match=re.escape("Can't compute inputs from scenarios: missing_input."),
    ):
        MDOScenarioAdapter(idf_scenario, [*inputs, "missing_input"], outputs)

    with pytest.raises(
        ValueError,
        match=re.escape("Can't compute outputs from scenarios: missing_output."),
    ):
        MDOScenarioAdapter(idf_scenario, inputs, [*outputs, "missing_output"])


def test_repr_str(idf_scenario) -> None:
    assert str(idf_scenario) == idf_scenario.name

    expected = [
        "MDOScenario",
        "   Disciplines: "
        "SobieskiAerodynamics SobieskiMission SobieskiPropulsion SobieskiStructure",
        "   MDO formulation: IDF",
    ]
    assert repr(idf_scenario) == "\n".join(expected)


def test_xdsm_filename(tmp_wd, idf_scenario) -> None:
    """Tests the export path dir for xdsm."""
    file_name = "my_xdsm.html"
    idf_scenario.xdsmize(file_name=file_name)
    assert Path(file_name).is_file()


@pytest.mark.parametrize(
    ("output_names", "observable_name", "expected"),
    [
        ("y_12", "", "y_12"),
        ("y_12", "foo", "foo"),
        (["y_12"], "", "y_12"),
        (["y_12", "y_23"], "", "y_12_y_23"),
    ],
)
def test_add_observable(mdf_scenario, output_names, observable_name, expected):
    """Test adding observables from discipline outputs."""
    mdf_scenario.add_observable(output_names, observable_name=observable_name)
    assert mdf_scenario.formulation.optimization_problem.observables[0].name == expected


def test_add_observable_not_available(
    mdf_scenario: MDOScenario,
) -> None:
    """Test adding an observable which is not available in any discipline.

    Args:
         mdf_scenario: A fixture for the MDOScenario.
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            "No discipline known by formulation MDF has all outputs named ['toto']."
        ),
    ):
        mdf_scenario.add_observable("toto")


def test_database_name(mdf_scenario) -> None:
    """Check the name of the database."""
    assert mdf_scenario.formulation.optimization_problem.database.name == "MDOScenario"


@patch("gemseo.core.execution_statistics.ExecutionStatistics.duration", new=1)
@pytest.mark.parametrize(
    ("is_enabled", "expected"), [(False, ""), (True, "(time: 0:00:00) ")]
)
def test_run_log(mdf_scenario, caplog, is_enabled, expected) -> None:
    """Check the log message of Scenario._run."""
    old_is_enabled = mdf_scenario.execution_statistics.is_enabled
    mdf_scenario.execution_statistics.is_enabled = is_enabled
    mdf_scenario._execute = lambda: None
    mdf_scenario.name = "ABC Scenario"
    mdf_scenario.execute(max_iter=1, algo_name="SLSQP")
    strings = [
        "*** Start ABC Scenario execution ***",
        f"*** End ABC Scenario execution {expected}***",
    ]
    for string in strings:
        assert string in caplog.text

    mdf_scenario.execution_statistics.is_enabled = old_is_enabled


def test_clear_history_before_run(mdf_scenario) -> None:
    """Check that clear_history_before_execute is correctly used in Scenario._run."""
    mdf_scenario.execute(algo_name="SLSQP", max_iter=1)
    assert len(mdf_scenario.formulation.optimization_problem.database) == 1

    def run_algorithm_mock() -> None:
        pass

    mdf_scenario._execute = run_algorithm_mock
    mdf_scenario.execute(algo_name="SLSQP", max_iter=1)
    assert len(mdf_scenario.formulation.optimization_problem.database) == 1

    mdf_scenario.clear_history_before_execute = True
    mdf_scenario.execute(algo_name="SLSQP", max_iter=1)
    assert len(mdf_scenario.formulation.optimization_problem.database) == 0


@pytest.mark.parametrize(
    ("activate", "text"),
    [
        (True, "Scenario Execution Statistics"),
        (False, "The discipline counters are disabled."),
    ],
)
def test_print_execution_metrics(mdf_scenario, caplog, activate, text) -> None:
    """Check the print of the execution metrics w.r.t."""
    activate_counters = ExecutionStatistics.is_enabled
    ExecutionStatistics.is_enabled = activate
    mdf_scenario.execute(algo_name="SLSQP", max_iter=1)
    mdf_scenario.print_execution_metrics()
    assert text in caplog.text
    ExecutionStatistics.is_enabled = activate_counters


def test_get_execution_metrics(mdf_scenario) -> None:
    """Check the string returned execution_metrics."""
    mdf_scenario.execute(algo_name="SLSQP", max_iter=1)
    expected = re.compile(
        r"""Scenario Execution Statistics
   Discipline: SobieskiPropulsion
      Executions number: 9
      Execution time: .* s
      Linearizations number: 1
   Discipline: SobieskiAerodynamics
      Executions number: 10
      Execution time: .* s
      Linearizations number: 1
   Discipline: SobieskiMission
      Executions number: 1
      Execution time: .* s
      Linearizations number: 1
   Discipline: SobieskiStructure
      Executions number: 10
      Execution time: .* s
      Linearizations number: 1
   Total number of executions calls: 30
   Total number of linearizations: 4"""
    )

    assert expected.match(str(mdf_scenario._BaseScenario__get_execution_metrics()))


def mocked_export_to_dataset(
    name: str | None = None,
    by_group: bool = True,
    categorize: bool = True,
    opt_naming: bool = True,
    export_gradients: bool = False,
) -> Dataset:
    """A mock for OptimizationProblem.to_dataset."""
    return (
        name,
        by_group,
        categorize,
        opt_naming,
        export_gradients,
    )


def test_export_to_dataset(mdf_scenario) -> None:
    """Check that to_dataset calls OptimizationProblem.to_dataset."""
    mdf_scenario.execute(algo_name="SLSQP", max_iter=1)
    mdf_scenario.to_dataset = mocked_export_to_dataset
    dataset = mdf_scenario.to_dataset(
        name=1, by_group=2, categorize=3, opt_naming=4, export_gradients=5
    )
    assert dataset == (1, 2, 3, 4, 5)


@pytest.fixture
def complex_step_scenario() -> MDOScenario:
    """The scenario to be used by test_complex_step."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)

    class MyDiscipline(Discipline):
        """The identity discipline f computing y = f(x) = x."""

        def __init__(self) -> None:
            super().__init__()
            self.io.input_grammar.update_from_names(["x"])
            self.io.output_grammar.update_from_names(["y"])

        def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
            self.io.data["y"] = self.io.data["x"]

    scenario = MDOScenario(
        [MyDiscipline()], "y", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.set_differentiation_method(scenario.DifferentiationMethod.COMPLEX_STEP)
    return scenario


@pytest.mark.parametrize("normalize_design_space", [False, True])
def test_complex_step(complex_step_scenario, normalize_design_space) -> None:
    """Check that complex step approximation works correctly."""
    complex_step_scenario.execute(
        algo_name="SLSQP", max_iter=10, normalize_design_space=normalize_design_space
    )

    assert complex_step_scenario.optimization_result.x_opt[0] == 0.0


@pytest.fixture
def sinus_use_case() -> tuple[AnalyticDiscipline, DesignSpace]:
    """The sinus discipline and its design space."""
    discipline = AnalyticDiscipline({"y": "sin(2*pi*x)"})
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return discipline, design_space


@pytest.mark.parametrize(
    ("maximize", "standardize", "expr", "val"),
    [
        (False, False, "minimize y(x)", -1.0),
        (False, True, "minimize y(x)", -1.0),
        (True, False, "maximize y(x)", 1.0),
        (True, True, "minimize -y(x)", -1.0),
    ],
)
def test_use_standardized_objective(
    sinus_use_case, maximize, standardize, expr, val, caplog
) -> None:
    """Check that the setter use_standardized_objective works correctly."""
    discipline, design_space = sinus_use_case
    scenario = MDOScenario(
        [discipline],
        "y",
        design_space,
        formulation_name="MDF",
        maximize_objective=maximize,
    )
    assert scenario.use_standardized_objective
    scenario.use_standardized_objective = standardize
    assert scenario.use_standardized_objective is standardize
    scenario.execute(algo_name="SLSQP", max_iter=10)
    assert expr in caplog.text
    assert f"Objective: {val}" in caplog.text
    assert f"obj={int(val)}" in caplog.text


@pytest.mark.parametrize(
    ("cast_default_inputs_to_complex", "expected_dtype"),
    [(True, complex128), (False, float64)],
)
def test_complex_casting(
    cast_default_inputs_to_complex, expected_dtype, mdf_scenario: MDOScenario
) -> None:
    """Check the automatic casting of default inputs when complex_step is selected.

    Args:
        cast_default_inputs_to_complex: Whether to cast the default inputs of the
            scenario's disciplines to complex.
        expected_dtype: The expected ``dtype`` after setting the differentiation method.
        mdf_scenario: A fixture for the MDOScenario.
    """
    for discipline in mdf_scenario.disciplines:
        for value in discipline.io.input_grammar.defaults.values():
            assert value.dtype == float64

    mdf_scenario.set_differentiation_method(
        mdf_scenario.DifferentiationMethod.COMPLEX_STEP,
        cast_default_inputs_to_complex=cast_default_inputs_to_complex,
    )
    for discipline in mdf_scenario.disciplines:
        for value in discipline.io.input_grammar.defaults.values():
            assert value.dtype == expected_dtype


@pytest.fixture
def scenario_with_non_float_variables() -> MDOScenario:
    """Create an ``MDOScenario`` from an ``AnalyticDiscipline`` with non-float inputs.

    Returns:
        The MDOScenario.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)

    discipline = AnalyticDiscipline({"y": "x"})
    discipline.io.input_grammar.update_from_names(["z"])
    discipline.io.input_grammar.update_from_names(["w"])
    discipline.io.input_grammar.defaults["z"] = "some_str"
    discipline.io.input_grammar.defaults["w"] = array(1, dtype=int64)

    return MDOScenario(
        [discipline], "y", design_space, formulation_name="DisciplinaryOpt"
    )


@pytest.mark.parametrize(
    ("cast_default_inputs_to_complex", "expected_dtype"),
    [(True, complex128), (False, float64)],
)
def test_complex_casting_with_non_float_variables(
    cast_default_inputs_to_complex, expected_dtype, scenario_with_non_float_variables
) -> None:
    """Test that the scenario will not cast non-float variables to complex.

    Args:
        cast_default_inputs_to_complex: Whether to cast the float default inputs of the
            scenario's disciplines to complex.
        expected_dtype: The expected ``dtype`` after setting the differentiation method.
        scenario_with_non_float_variables: Fixture that returns an ``MDOScenario`` with
            an AnalyticDiscipline that has integer and string inputs.
    """
    scenario_with_non_float_variables.set_differentiation_method(
        scenario_with_non_float_variables.DifferentiationMethod.COMPLEX_STEP,
        cast_default_inputs_to_complex=cast_default_inputs_to_complex,
    )

    assert (
        scenario_with_non_float_variables.formulation.design_space._current_value[
            "x"
        ].dtype
        == complex128
    )
    assert (
        scenario_with_non_float_variables.disciplines[0].default_input_data["x"].dtype
        == expected_dtype
    )
    assert isinstance(
        scenario_with_non_float_variables.disciplines[0].default_input_data["z"], str
    )
    assert (
        scenario_with_non_float_variables.disciplines[0].default_input_data["w"].dtype
        == int64
    )


def test_check_disciplines() -> None:
    """Test that an exception is raised when two disciplines compute the same output."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)

    discipline_1 = AnalyticDiscipline({"y": "x"}, name="foo")
    discipline_2 = AnalyticDiscipline({"y": "x + 1"}, name="bar")
    with pytest.raises(
        ValueError,
        match="Two disciplines, among "
        f"which ({discipline_1.name}|({discipline_2.name})), "
        "compute the same outputs: {'y'}",
    ):
        MDOScenario(
            [discipline_1, discipline_2],
            "y",
            design_space,
            formulation_name="DisciplinaryOpt",
        )


@pytest.fixture
def identity_scenario() -> MDOScenario:
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return MDOScenario(
        [AnalyticDiscipline({"y": "x"})],
        "y",
        design_space,
        formulation_name="DisciplinaryOpt",
    )


@pytest.mark.parametrize(
    ("constraint_type", "constraint_name", "value", "positive", "expected"),
    [
        ("eq", "", 0.0, False, ["y", "", "y(x) == 0.0", "y(x) == 0.0"]),
        (
            "eq",
            "cstr",
            0.0,
            False,
            ["cstr", "", "y(x) == 0.0", "cstr: y(x) == 0.0"],
        ),
        (
            "eq",
            "",
            1.0,
            False,
            ["[y-1.0]", "y(x)-1.0", "y(x)-1.0 == 0.0", "y(x) == 1.0"],
        ),
        (
            "eq",
            "",
            -1.0,
            False,
            ["[y+1.0]", "y(x)+1.0", "y(x)+1.0 == 0.0", "y(x) == -1.0"],
        ),
        (
            "eq",
            "cstr",
            1.0,
            False,
            ["cstr", "y(x)-1.0", "y(x)-1.0 == 0.0", "cstr: y(x) == 1.0"],
        ),
        (
            "eq",
            "cstr",
            -1.0,
            False,
            ["cstr", "y(x)+1.0", "y(x)+1.0 == 0.0", "cstr: y(x) == -1.0"],
        ),
        ("ineq", "", 0.0, False, ["y", "", "y(x) <= 0.0", "y(x) <= 0.0"]),
        (
            "ineq",
            "cstr",
            0.0,
            False,
            ["cstr", "", "y(x) <= 0.0", "cstr: y(x) <= 0.0"],
        ),
        (
            "ineq",
            "",
            1.0,
            False,
            ["[y-1.0]", "y(x)-1.0", "y(x)-1.0 <= 0.0", "y(x) <= 1.0"],
        ),
        (
            "ineq",
            "",
            -1.0,
            False,
            ["[y+1.0]", "y(x)+1.0", "y(x)+1.0 <= 0.0", "y(x) <= -1.0"],
        ),
        (
            "ineq",
            "cstr",
            1.0,
            False,
            ["cstr", "y(x)-1.0", "y(x)-1.0 <= 0.0", "cstr: y(x) <= 1.0"],
        ),
        (
            "ineq",
            "cstr",
            -1.0,
            False,
            ["cstr", "y(x)+1.0", "y(x)+1.0 <= 0.0", "cstr: y(x) <= -1.0"],
        ),
        ("ineq", "", 0.0, True, ["-y", "-y(x)", "-y(x) <= 0.0", "y(x) >= 0.0"]),
        (
            "ineq",
            "cstr",
            0.0,
            True,
            ["cstr", "-y(x)", "-y(x) <= 0.0", "cstr: y(x) >= 0.0"],
        ),
        (
            "ineq",
            "",
            1.0,
            True,
            ["-[y-1.0]", "-(y(x)-1.0)", "-(y(x)-1.0) <= 0.0", "y(x) >= 1.0"],
        ),
        (
            "ineq",
            "",
            -1.0,
            True,
            ["-[y+1.0]", "-(y(x)+1.0)", "-(y(x)+1.0) <= 0.0", "y(x) >= -1.0"],
        ),
        (
            "ineq",
            "cstr",
            1.0,
            True,
            ["cstr", "-(y(x)-1.0)", "-(y(x)-1.0) <= 0.0", "cstr: y(x) >= 1.0"],
        ),
        (
            "ineq",
            "cstr",
            -1.0,
            True,
            ["cstr", "-(y(x)+1.0)", "-(y(x)+1.0) <= 0.0", "cstr: y(x) >= -1.0"],
        ),
    ],
)
def test_constraint_representation(
    identity_scenario, constraint_type, constraint_name, value, positive, expected
) -> None:
    """"""
    identity_scenario.add_constraint(
        "y",
        constraint_type=constraint_type,
        constraint_name=constraint_name,
        value=value,
        positive=positive,
    )
    constraints = identity_scenario.formulation.optimization_problem.constraints[-1]
    assert constraints.name == expected[0]
    assert constraints.expr == expected[1]
    assert constraints.default_repr == expected[2]
    assert constraints.special_repr == expected[3]


def test_lib_serialization(tmp_wd, mdf_scenario) -> None:
    """Test the serialization of an MDOScenario with an instantiated opt_lib.

    Args:
        mdf_scenario: A fixture for the MDOScenario.
    """
    mdf_scenario.execute(algo_name="SLSQP", max_iter=1)
    mdf_scenario.formulation.optimization_problem.reset(
        database=False, design_space=False
    )

    with open("scenario.pkl", "wb") as file:
        pickle.dump(mdf_scenario, file)

    with open("scenario.pkl", "rb") as file:
        pickled_scenario = pickle.load(file)

    pickled_scenario.execute(algo_name="SLSQP", max_iter=1)


def test_get_result(mdf_scenario) -> None:
    """Check get_result."""
    assert mdf_scenario.get_result() is None

    mdf_scenario.execute(algo_name="SLSQP", max_iter=1)
    assert mdf_scenario.get_result().design_variable_names_to_values

    with pytest.raises(
        ImportError,
        match=r"The class foo is not available; the available ones are: .*",
    ):
        mdf_scenario.get_result("foo")


@pytest.fixture(params=[True, False])
def full_linear(request):
    """Whether the generated problem should be linear."""
    return request.param


@pytest.fixture
def scenario_for_linear_check(full_linear):
    """MDOScenario for linear check."""
    my_disc = AnalyticDiscipline({"f": "x1+ x2**2"})
    my_disc.io.input_grammar.defaults = {"x1": array([0.5]), "x2": array([0.5])}
    my_disc.io.set_linear_relationships(["x1"], ["f"])
    ds = DesignSpace()
    ds.add_variable("x1", 1, lower_bound=0.0, upper_bound=1.0, value=0.5)
    if not full_linear:
        ds.add_variable("x2", 1, lower_bound=0.0, upper_bound=1.0, value=0.5)
    return create_scenario(my_disc, "f", ds, formulation_name="DisciplinaryOpt")


def test_function_problem_type(scenario_for_linear_check, full_linear) -> None:
    """Test that function and problem are consistent with declaration."""
    optimization_problem = scenario_for_linear_check.formulation.optimization_problem
    if not full_linear:
        assert isinstance(
            optimization_problem.objective,
            FunctionFromDiscipline,
        )
        assert not optimization_problem.is_linear
    else:
        assert isinstance(optimization_problem.objective, MDOLinearFunction)
        assert scenario_for_linear_check.formulation.optimization_problem.is_linear


class MyDisc(Discipline):
    """A discipline to manage different type of data."""

    default_cache_type = Discipline.CacheType.NONE
    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(self):
        super().__init__()
        self.io.input_grammar.update_from_data({"x_float": array([0.0, 0.0])})
        self.io.input_grammar.update_from_data({"x_int": array([0])})
        self.io.output_grammar.update_from_data({"y1": array([0.0])})
        self.io.output_grammar.update_from_data({"y2": array([0.0, 0.0])})
        self.io.output_grammar.update_from_data({"name": array(["foo"])})

    def _run(self, input_data: StrKeyMapping):
        return {
            "y1": self.io.get_input_data()["x_int"],
            "y2": self.io.get_input_data()["x_float"],
            "name": array(["foo"]),
        }


def test_scenario_to_dataset(tmp_wd):
    """Test to_dataset method when there are different data types."""
    design_space = DesignSpace()
    design_space.add_variable("x_float", size=2)
    design_space.add_variable("x_int", type_=DesignSpace.DesignVariableType.INTEGER)

    scenario = DOEScenario(
        [MyDisc()], "y1", design_space, formulation_name="DisciplinaryOpt"
    )
    scenario.add_observable("y2")
    scenario.add_observable("name")

    scenario.execute(
        algo_name="CustomDOE", samples=array([[0.0, 0.0, 1], [3.0, 3.0, 5]])
    )
    dataset = scenario.to_dataset(name="foo", opt_naming=False)

    reference_dataset = Dataset()
    reference_dataset.add_variable("x_float", [0.0, 3.0], "inputs", components=0)
    reference_dataset.add_variable("x_float", [0.0, 3.0], "inputs", components=1)
    reference_dataset.add_variable("x_int", [1, 5], "inputs", components=0)
    reference_dataset.add_variable("name", "foo", "outputs", components=0)
    reference_dataset.add_variable("y1", [1, 5], "outputs", components=0)
    reference_dataset.add_variable("y2", [0.0, 3.0], "outputs", components=0)
    reference_dataset.add_variable("y2", [0.0, 3.0], "outputs", components=1)
    assert_frame_equal(dataset, reference_dataset, check_dtype=False)


@pytest.mark.parametrize(
    ("use_doe_first", "expected"),
    [
        (False, [array([1.0]), array([-0.5]), array([0.0]), array([0.123])]),
        (
            True,
            [
                array([0.123]),
                array([0.41533333]),
                array([1.11022302e-16]),
                array([0.0]),
            ],
        ),
    ],
)
def test_opt_and_doe(use_doe_first, expected):
    """Check the execution of a scenario with an optimizer and a DOE."""
    discipline = AnalyticDiscipline({"f": "x**2"})
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-0.5, upper_bound=1.0, value=1.0)
    scenario = MDOScenario(
        [discipline], "f", design_space, formulation_name="DisciplinaryOpt"
    )
    if use_doe_first:
        scenario.execute(algo_name="CustomDOE", samples=array([[0.123]]))
        scenario.execute(algo_name="SLSQP", max_iter=3)
    else:
        scenario.execute(algo_name="SLSQP", max_iter=3)
        scenario.execute(algo_name="CustomDOE", samples=array([[0.123]]))

    x = scenario.formulation.optimization_problem.database.get_x_vect_history()
    assert_almost_equal(x, expected)
