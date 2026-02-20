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
#      :author: Francois Gallard, Gilberto Ruiz
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pickle

import pytest
from numpy import array
from numpy import ndarray  # noqa: TC002
from pydantic import ValidationError

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.doe.scipy.settings.lhs import LHS_Settings
from gemseo.core.discipline import Discipline
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.formulations.factory import MDO_FORMULATION_FACTORY
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiAerodynamicsSG
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiMissionSG
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiPropulsionSG
from gemseo.problems.mdo.sobieski._disciplines_sg import SobieskiStructureSG
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo import MDOScenario


def build_mdo_scenario(
    formulation_name: str,
    grammar_type: Discipline.GrammarType = Discipline.GrammarType.JSON,
) -> MDOScenario:
    """Build the MDO scenario for SSBJ.

    Args:
        formulation_name: The name of the MDO formulation.
        grammar_type: The grammar type.

    Returns:
        The MDO scenario.
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
        design_space,
        formulation_settings=MDO_FORMULATION_FACTORY.get_class(
            formulation_name
        ).settings_class(),
    )
    scenario.add_objective("y_4", minimize=False)
    return scenario


@pytest.fixture
def mdf_variable_grammar_doe_scenario(request):
    """Return an MDOScenario with MDF formulation and custom grammar.

    Args:
        request: An auxiliary variable to retrieve the grammar type with
            pytest.mark.parametrize and the option `indirect=True`.
    """
    return build_mdo_scenario("MDF", request.param)


@pytest.mark.usefixtures("tmp_wd")
@pytest.mark.skip_under_windows
def test_parallel_doe_hdf_cache(caplog) -> None:
    disciplines = create_discipline([
        "SobieskiStructure",
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
    ])
    path = "cache.h5"
    for disc in disciplines:
        disc.set_cache(disc.CacheType.HDF5, hdf_file_path=path)

    scenario = create_scenario(
        disciplines,
        "y_4",
        SobieskiDesignSpace(),
        formulation_name="DisciplinaryOpt",
        maximize_objective=True,
    )

    n_samples = 10
    scenario.execute(LHS_Settings(n_samples=n_samples, n_processes=2))
    scenario.print_execution_metrics()
    assert len(scenario.formulation.problem.database) == n_samples
    for disc in disciplines:
        assert len(disc.cache) == n_samples


@pytest.mark.parametrize(
    "mdf_variable_grammar_doe_scenario",
    [Discipline.GrammarType.SIMPLE, Discipline.GrammarType.JSON],
    indirect=True,
)
def test_doe_scenario(mdf_variable_grammar_doe_scenario) -> None:
    """Test the execution of a MDOScenario with different grammars.

    Args:
        mdf_variable_grammar_doe_scenario: The MDOScenario.
    """
    n_samples = 10
    mdf_variable_grammar_doe_scenario.execute(
        LHS_Settings(n_samples=n_samples, n_processes=1)
    )
    mdf_variable_grammar_doe_scenario.print_execution_metrics()
    assert (
        len(mdf_variable_grammar_doe_scenario.formulation.problem.database) == n_samples
    )


@pytest.fixture(scope="module")
def unit_design_space():
    """A unit design space with x as variable."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return design_space


@pytest.fixture(scope="module")
def double_discipline():
    """An analytic discipline that doubles its input."""
    return AnalyticDiscipline({"y": "2*x"}, name="func")


@pytest.fixture
def doe_scenario(unit_design_space, double_discipline) -> MDOScenario:
    """A simple MDO scenario not yet executed.

    Args:
        unit_design_space: A unit design space with x as a variable.
        double_discipline: An analytic discipline that doubles its input.

    Minimize y=func(x)=2x over [0,1].
    """
    scenario = MDOScenario(
        [double_discipline],
        unit_design_space,
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    scenario.add_objective("y")
    return scenario


def test_validation_exception(doe_scenario) -> None:
    """Check that an exception is raised when a setting is unknown.

    Args:
        doe_scenario: A simple DOE scenario.
    """
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        doe_scenario.execute(
            CustomDOE_Settings(samples=array([[1.0]]), unknown_setting=1)
        )


def f_sellar_1(x_1: ndarray, y_2: ndarray, x_shared: ndarray) -> ndarray:
    """Function for discipline 1."""
    if x_1 == 0.0:
        msg = "Undefined"
        raise ValueError(msg)

    y_1 = (x_shared[0] ** 2 + x_shared[1] + x_1 - 0.2 * y_2) ** 0.5
    return y_1  # noqa: RET504


@pytest.mark.parametrize("use_threading", [True, False])
def test_exception_mda_jacobi(
    caplog, use_threading, sellar_disciplines, enable_discipline_statistics
) -> None:
    """Check that a MDO scenario does not crash with a ValueError and MDAJacobi.

    Args:
        caplog: Fixture to access and control log capturing.
        use_threading: Whether to use threading in the MDAJacobi.
    """
    sellar_disciplines = list(sellar_disciplines)
    sellar_disciplines[0] = create_discipline("AutoPyDiscipline", py_func=f_sellar_1)

    scenario = MDOScenario(
        sellar_disciplines,
        SellarDesignSpace(),
        "obj",
        formulation_settings=MDF_Settings(
            main_mda_name="MDAChain",
            main_mda_settings={
                "inner_mda_name": "MDAJacobi",
                "use_threading": use_threading,
                "n_processes": 2,
            },
        ),
    )
    scenario.add_objective("obj")
    scenario.execute(CustomDOE_Settings(samples=array([[0.0, 0.0, -10.0, 0.0]])))

    assert sellar_disciplines[2].execution_statistics.n_executions == 0
    assert "Undefined" in caplog.text


def test_other_exceptions_caught(caplog) -> None:
    """Check that exceptions that are not ValueErrors are not re-raised.

    Args:
        caplog: Fixture to access and control log capturing.
    """
    discipline = AnalyticDiscipline({"y": "1/x"}, name="func")
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    scenario = MDOScenario(
        [discipline],
        design_space,
        formulation_settings=MDF_Settings(main_mda_name="MDAJacobi"),
    )
    scenario.add_objective("y")
    with pytest.raises(InvalidDataError):
        scenario.execute(CustomDOE_Settings(samples=array([[0.0]])))
    assert "0.0 cannot be raised to a negative power" in caplog.text


def test_export_to_dataset_with_repeated_inputs() -> None:
    """Check the export of the database with repeated inputs."""
    discipline = AnalyticDiscipline({"obj": "2*dv"}, "f")
    design_space = DesignSpace()
    design_space.add_variable("dv")
    scenario = MDOScenario(
        [discipline], design_space, formulation_settings=DisciplinaryOpt_Settings()
    )
    scenario.add_objective("obj")
    samples = array([[1.0], [2.0], [1.0]])
    scenario.execute(CustomDOE_Settings(samples=samples))
    dataset = scenario.to_dataset()
    assert (dataset.get_view(variable_names="dv").to_numpy() == samples).all()
    assert (dataset.get_view(variable_names="obj").to_numpy() == samples * 2).all()


def test_export_to_dataset_normalized_integers() -> None:
    """Check the export of the database with normalized integers."""
    discipline = AnalyticDiscipline({"obj": "2*dv"}, "f")
    design_space = DesignSpace()
    design_space.add_variable("dv", type_="integer", lower_bound=1, upper_bound=10)
    scenario = MDOScenario(
        [discipline], design_space, formulation_settings=DisciplinaryOpt_Settings()
    )
    scenario.add_objective("obj")
    samples = array([[1], [2], [10]])
    scenario.execute(CustomDOE_Settings(samples=samples))
    dataset = scenario.to_dataset()
    assert (dataset.get_view(variable_names="dv").to_numpy() == samples).all()
    assert (dataset.get_view(variable_names="obj").to_numpy() == samples * 2).all()


def test_lib_serialization(tmp_wd, doe_scenario) -> None:
    """Test the serialization of a MDOScenario with an instantiated BaseDOELibrary.

    Args:
        tmp_wd: Fixture to move into a temporary work directory.
        doe_scenario: A simple DOE scenario.
    """
    doe_scenario.execute(CustomDOE_Settings(samples=array([[1.0]])))

    doe_scenario.formulation.problem.reset(database=False, design_space=False)

    with open("doe.pkl", "wb") as file:
        pickle.dump(doe_scenario, file)

    with open("doe.pkl", "rb") as file:
        pickled_scenario = pickle.load(file)

    assert (
        pickled_scenario._EvaluationScenario__algorithm_settings
        == doe_scenario._EvaluationScenario__algorithm_settings
    )

    pickled_scenario.execute(CustomDOE_Settings(samples=array([[0.5]])))

    assert pickled_scenario.formulation.problem.database.get_function_value(
        "y", array([0.5])
    ) == array([1.0])
    assert pickled_scenario.formulation.problem.database.get_function_value(
        "y", array([1.0])
    ) == array([2.0])


other_doe_scenario = doe_scenario


@pytest.mark.parametrize(
    ("samples_1", "samples_2", "reset_iteration_counters", "expected"),
    [
        (array([[0.5]]), array([[0.25], [0.75]]), True, 3),
        (array([[0.5]]), array([[0.25], [0.75]]), False, 2),
        (array([[0.5]]), array([[0.25]]), True, 2),
        (array([[0.5]]), array([[0.25]]), False, 1),
        (array([[0.5], [0.75]]), array([[0.25]]), True, 3),
        (array([[0.5], [0.75]]), array([[0.25]]), False, 2),
    ],
)
def test_partial_execution_from_backup(
    tmp_wd,
    doe_scenario,
    other_doe_scenario,
    samples_1,
    samples_2,
    reset_iteration_counters,
    expected,
) -> None:
    """Test the execution of a MDOScenario from a backup.

    Args:
        doe_scenario: A simple DOE scenario.
        other_doe_scenario: A different instance of a simple DOE scenario.
        samples_1: The samples for the first execution.
        samples_2: The samples for the second execution.
        reset_iteration_counters: Whether to reset the iteration counters from the
            previous execution of the scenario to 0 before executing it again.
        expected: The expected database length.
    """
    doe_scenario.set_backup_settings("backup.h5")
    doe_scenario.execute(CustomDOE_Settings(samples=samples_1))
    other_doe_scenario.set_backup_settings("backup.h5", load=True)
    other_doe_scenario.execute(
        CustomDOE_Settings(
            samples=samples_2, reset_iteration_counters=reset_iteration_counters
        )
    )
    assert len(other_doe_scenario.formulation.problem.database) == expected


def test_scenario_without_initial_design_value() -> None:
    """Check that a MDOScenario can work without initial design value."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    discipline = AnalyticDiscipline({"y": "x"})
    discipline.io.input_grammar.defaults = {}
    scenario = MDOScenario(
        [discipline], design_space, formulation_settings=MDF_Settings()
    )
    scenario.add_objective("y")
    scenario.execute(LHS_Settings(n_samples=3))
    assert len(scenario.formulation.problem.database) == 3
