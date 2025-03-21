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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from numpy import array
from numpy import cos
from numpy import linspace
from numpy import newaxis
from numpy import pi as np_pi
from numpy import sin
from numpy.testing import assert_equal

from gemseo import DatasetClassName
from gemseo import compute_doe
from gemseo import configure
from gemseo import configure_logger
from gemseo import create_benchmark_dataset
from gemseo import create_cache
from gemseo import create_dataset
from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_mda
from gemseo import create_parameter_space
from gemseo import create_scalable
from gemseo import create_scenario
from gemseo import create_surrogate
from gemseo import execute_algo
from gemseo import execute_post
from gemseo import generate_coupling_graph
from gemseo import generate_n2_plot
from gemseo import generate_xdsm
from gemseo import get_algorithm_options_schema
from gemseo import get_available_caches
from gemseo import get_available_disciplines
from gemseo import get_available_doe_algorithms
from gemseo import get_available_formulations
from gemseo import get_available_mdas
from gemseo import get_available_opt_algorithms
from gemseo import get_available_post_processings
from gemseo import get_available_scenario_types
from gemseo import get_available_surrogates
from gemseo import get_discipline_inputs_schema
from gemseo import get_discipline_options_defaults
from gemseo import get_discipline_options_schema
from gemseo import get_discipline_outputs_schema
from gemseo import get_formulation_options_schema
from gemseo import get_formulation_sub_options_schema
from gemseo import get_formulations_options_defaults
from gemseo import get_formulations_sub_options_defaults
from gemseo import get_mda_options_schema
from gemseo import get_post_processing_options_schema
from gemseo import get_scenario_differentiation_modes
from gemseo import get_scenario_inputs_schema
from gemseo import get_scenario_options_schema
from gemseo import get_surrogate_options_schema
from gemseo import import_database
from gemseo import import_discipline
from gemseo import monitor_scenario
from gemseo import print_configuration
from gemseo import sample_disciplines
from gemseo import wrap_discipline_in_job_scheduler
from gemseo import write_design_space
from gemseo.algos.base_driver_library import BaseDriverLibrary
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.algos.problem_function import ProblemFunction
from gemseo.core.discipline import Discipline
from gemseo.core.execution_statistics import ExecutionStatistics
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.datasets.io_dataset import IODataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mda.base_mda import BaseMDA
from gemseo.post._graph_view import GraphView
from gemseo.post.opt_history_view import OptHistoryView
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from gemseo.scenarios.backup_settings import BackupSettings
from gemseo.scenarios.base_scenario import BaseScenario
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.utils.logging_tools import LOGGING_SETTINGS
from gemseo.utils.logging_tools import MultiLineStreamHandler
from gemseo.utils.pickle import to_pickle
from gemseo.utils.xdsm import XDSM
from gemseo.utils.xdsmizer import XDSMizer

if TYPE_CHECKING:
    from gemseo.scenarios.mdo_scenario import MDOScenario


class Observer:
    def __init__(self) -> None:
        self.status_changes = 0

    def update(self, atom) -> None:
        self.status_changes += 1


@pytest.fixture(scope="module")
def scenario() -> MDOScenario:
    """An MDO scenario after execution."""
    scenario = create_scenario(
        create_discipline("SobieskiMission"),
        "y_4",
        SobieskiDesignSpace(),
        formulation_name="DisciplinaryOpt",
    )
    scenario.execute(algo_name="SLSQP", max_iter=10)
    return scenario


@pytest.fixture(scope="module")
def sobieski_disciplines():
    """The Sobieski disciplines."""
    return create_discipline([
        "SobieskiMission",
        "SobieskiAerodynamics",
        "SobieskiStructure",
        "SobieskiPropulsion",
    ])


def test_generate_n2_plot(tmp_wd, sobieski_disciplines) -> None:
    """Test the n2 plot with the Sobieski problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    file_path = "n2.png"
    generate_n2_plot(sobieski_disciplines, file_path, fig_size=(5, 5))
    assert Path(file_path).exists()


@pytest.mark.parametrize("full", [False, True])
def test_generate_coupling_graph(tmp_wd, full, sobieski_disciplines) -> None:
    """Test the coupling graph with the Sobieski problem."""
    # TODO: reuse data and checks from test_dependency_graph
    file_path = "coupl.pdf"
    graph_view = generate_coupling_graph(sobieski_disciplines, file_path, full)
    assert isinstance(graph_view, GraphView)
    assert Path(file_path).exists()
    assert Path("coupl.dot").exists()


@pytest.mark.parametrize("full", [False, True])
def test_generate_coupling_graph_without_saving(
    tmp_wd, full, sobieski_disciplines
) -> None:
    """Check the coupling graph without saving the files."""
    graph_view = generate_coupling_graph(sobieski_disciplines, file_path="", full=full)
    assert isinstance(graph_view, GraphView)
    assert not list(tmp_wd.iterdir())


def test_get_algorithm_options_schema() -> None:
    """Test that all available options are printed."""
    schema_dict = get_algorithm_options_schema("SLSQP")
    assert "properties" in schema_dict
    assert len(schema_dict["properties"]) == 22

    schema_json = get_algorithm_options_schema("SLSQP", output_json=True)
    out_dict = json.loads(schema_json)
    for key, val in schema_dict.items():
        assert key in out_dict
        assert out_dict[key] == val

    with pytest.raises(
        ValueError, match=re.escape("Algorithm named unknown is not available.")
    ):
        get_algorithm_options_schema("unknown")

    get_algorithm_options_schema("SLSQP", pretty_print=True)


def test_get_post_processing_options_schema() -> None:
    """Test that all available options are printed."""
    schema_dict = get_post_processing_options_schema("OptHistoryView")
    assert "properties" in schema_dict
    assert len(schema_dict["properties"]) == 11

    schema_json = get_post_processing_options_schema("OptHistoryView", output_json=True)
    out_dict = json.loads(schema_json)
    for key, val in schema_dict.items():
        assert key in out_dict
        assert out_dict[key] == val

    get_post_processing_options_schema("OptHistoryView", pretty_print=True)


def test_get_surrogate_options_schema() -> None:
    """Test that the surrogate options schema is printed."""
    get_surrogate_options_schema("LinearRegressor")
    get_surrogate_options_schema("LinearRegressor", pretty_print=True)


def test_create_scenario_and_monitor() -> None:
    """Test the creation of a scenario from the SobieskiMission discipline."""
    create_scenario(
        create_discipline("SobieskiMission"),
        "y_4",
        SobieskiDesignSpace(),
        formulation_name="DisciplinaryOpt",
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Unknown scenario type: unknown, use one of : 'MDO' or 'DOE'."),
    ):
        create_scenario(
            create_discipline("SobieskiMission"),
            "y_4",
            SobieskiDesignSpace(),
            formulation_name="DisciplinaryOpt",
            scenario_type="unknown",
        )


def test_monitor_scenario() -> None:
    """Test the scenario monitoring API method."""
    scenario = create_scenario(
        create_discipline("SobieskiMission"),
        "y_4",
        SobieskiDesignSpace(),
        formulation_name="DisciplinaryOpt",
    )

    observer = Observer()
    monitor_scenario(scenario, observer)

    scenario.execute(algo_name="SLSQP", max_iter=10)
    assert (
        observer.status_changes
        >= 2 * scenario.formulation.optimization_problem.objective.n_calls
    )


@pytest.mark.parametrize("obj_type", [BaseScenario, str, Path])
def test_execute_post(scenario, obj_type, tmp_wd) -> None:
    """Test the API method to call the post-processing factory.

    Args:
        scenario: An MDO scenario after execution.
        obj_type: The type of the object to post-process.
        tmp_wd: Fixture to move into a temporary directory.
    """
    if obj_type is BaseScenario:
        obj = scenario
    else:
        file_name = "results.hdf5"
        scenario.save_optimization_history(file_name)
        obj = file_name if obj_type is str else Path(file_name)

    post = execute_post(obj, post_name="OptHistoryView", save=False, show=False)
    assert isinstance(post, OptHistoryView)


def test_execute_post_type_error(scenario) -> None:
    """Test the method execute_post with a wrong typed argument."""
    with pytest.raises(TypeError, match=f"Cannot post process type: {int}"):
        execute_post(1234, post_name="OptHistoryView")


def test_create_doe_scenario() -> None:
    """Test the creation of a DOE scenario."""
    create_scenario(
        create_discipline("SobieskiMission"),
        "y_4",
        SobieskiDesignSpace(),
        formulation_name="DisciplinaryOpt",
        scenario_type="DOE",
    )


@pytest.mark.parametrize(
    ("formulation_name", "opts", "expected"),
    [
        (
            "MDF",
            {"main_mda_name": "MDAJacobi"},
            {"acceleration_method", "n_processes", "use_threading"},
        ),
        (
            "BiLevel",
            {"main_mda_name": "MDAGaussSeidel"},
            {"over_relaxation_factor"},
        ),
        ("DisciplinaryOpt", {}, None),
        ("IDF", {}, None),
    ],
)
def test_get_formulation_sub_options_schema(formulation_name, opts, expected) -> None:
    """Check that the sub options schema is recovered for different formulations.

    Args:
        formulation_name: The name of the formulation to test.
        opts: The options for the formulation.
        expected: The expected result.
    """
    sub_opts_schema = get_formulation_sub_options_schema(formulation_name, **opts)

    if formulation_name in {"BiLevel", "MDF"}:
        props = sub_opts_schema["properties"]
        assert expected.issubset(set(props))
    else:
        assert not sub_opts_schema


@pytest.mark.parametrize(
    ("formulation_name", "opts"),
    [
        (
            "MDF",
            {"main_mda_name": "MDAJacobi"},
        ),
        (
            "BiLevel",
            {"main_mda_name": "MDAGaussSeidel"},
        ),
        ("DisciplinaryOpt", {}),
        ("IDF", {}),
    ],
)
def test_get_formulation_sub_options_schema_print(
    capfd, formulation_name, opts
) -> None:
    """Check that the sub options schema is printed for different formulations.

    Args:
        capfd: Fixture capture outputs sent to ``stdout`` and
            ``stderr``.
        formulation_name: The name of the formulation to test.
        opts: The options for the formulation.
    """
    # A pattern for table headers.
    expected = re.compile(
        r"\+-+\+-+\+-+\+$\n\|\s+Name\s+\|\s+Description\s+\|\s+Type\s+\|$\n",
        re.MULTILINE,
    )
    schema = get_formulation_sub_options_schema(
        formulation_name, pretty_print=True, **opts
    )
    out, err = capfd.readouterr()
    assert not err
    if schema:
        assert bool(re.search(expected, out))


def test_get_scenario_inputs_schema() -> None:
    """Check that the scenario inputs schema is retrieved correctly."""
    aero = create_discipline(["SobieskiAerodynamics"])
    design_space = SobieskiDesignSpace()
    sc_aero = create_scenario(
        aero, "y_24", design_space.filter("x_2"), formulation_name="DisciplinaryOpt"
    )

    schema = get_scenario_inputs_schema(sc_aero)
    assert "algo_name" in schema["properties"]
    assert "algo_settings" in schema["properties"]

    get_scenario_inputs_schema(sc_aero, pretty_print=True)


def test_exec_algo() -> None:
    """Test the execution of an algorithm with the Rosenbrock problem."""
    problem = Rosenbrock()
    sol = execute_algo(problem, algo_name="L-BFGS-B", max_iter=200)
    assert abs(sol.f_opt) < 1e-8

    sol = execute_algo(problem, algo_name="LHS", algo_type="doe", n_samples=200)
    assert abs(sol.f_opt) < 1e-8

    with pytest.raises(
        ValueError,
        match="Unknown algo type: unknown_algo, please use 'doe' or 'opt' !",
    ):
        execute_algo(problem, algo_name="LHS", algo_type="unknown_algo", n_samples=200)


def test_get_scenario_options_schema() -> None:
    """Check that the scenario options schema is retrieved correctly."""
    schema = get_scenario_options_schema("MDO")
    assert "name" in schema["properties"]

    with pytest.raises(ValueError, match="Unknown scenario type UnknownType"):
        get_scenario_options_schema("UnknownType")

    get_scenario_options_schema("MDO", pretty_print=True)


def test_get_mda_options_schema() -> None:
    """Check that the mda options schema are retrieved correctly."""
    schema = get_mda_options_schema("MDAJacobi")
    assert "name" in schema["properties"]

    get_mda_options_schema("MDAJacobi", pretty_print=True)


def test_get_available_opt_algorithms() -> None:
    """Check that the optimization algorithms are retrieved correctly."""
    algos = get_available_opt_algorithms()
    assert "SLSQP" in algos
    assert "L-BFGS-B" in algos
    assert "TNC" in algos


def test_get_available_doe_algorithms() -> None:
    """Test that the doe algorithms are retrieved correctly."""
    algos = get_available_doe_algorithms()
    assert "LHS" in algos
    assert "PYDOE_FULLFACT" in algos


def test_get_available_formulations() -> None:
    """Test that the available formulations are retrieved correctly."""
    formulations = get_available_formulations()
    assert "MDF" in formulations
    assert "DisciplinaryOpt" in formulations
    assert "IDF" in formulations


def test_get_available_post_processings() -> None:
    """Test that the available post-processing methods are retrieved correctly."""
    post_processors = get_available_post_processings()
    assert "OptHistoryView" in post_processors
    assert "RadarChart" in post_processors


def test_get_available_surrogates() -> None:
    """Test that the available surrogates are retrieved correctly."""
    surrogates = get_available_surrogates()
    assert "RBFRegressor" in surrogates
    assert "LinearRegressor" in surrogates


def test_get_available_disciplines() -> None:
    """Test that the available disciplines are retrieved correctly."""
    disciplines = get_available_disciplines()
    assert "SobieskiMission" in disciplines
    assert "Sellar1" in disciplines


def test_create_discipline() -> None:
    """Test that API method creates a discipline properly.

    Test exceptions when the options dictionary does not follow the specified json
    grammar.
    """
    options = {
        "dtype": "float64",
        "linearization_mode": "auto",
        "cache_type": "SimpleCache",
    }
    miss = create_discipline("SobieskiMission", **options)
    miss.execute()
    assert isinstance(miss, Discipline)

    options_fail = {
        "dtype": "float64",
        "linearization_mode": "finite_differences",
        "cache_type": Discipline.CacheType.SIMPLE,
    }

    msg = (
        "data.linearization_mode must be one of "
        r"\['auto', 'direct', 'reverse', 'adjoint'\]"
    )
    with pytest.raises(InvalidDataError, match=msg):
        create_discipline("SobieskiMission", **options_fail)


def test_create_surrogate() -> None:
    """Test the creation of a surrogate discipline."""
    disc = SobieskiMission()
    input_names = ["y_24", "y_34"]
    disc.set_cache(disc.CacheType.MEMORY_FULL)
    design_space = SobieskiDesignSpace()
    design_space.filter(input_names)
    doe = DOEScenario([disc], "y_4", design_space, formulation_name="DisciplinaryOpt")
    doe.execute(algo_name="PYDOE_FULLFACT", n_samples=10)
    surr = create_surrogate(
        "RBFRegressor",
        disc.cache.to_dataset(),
        input_names=["y_24", "y_34"],
    )
    outs = surr.execute({"y_24": array([1.0]), "y_34": array([1.0])})

    assert outs["y_4"] > 0.0


def test_create_scalable() -> None:
    """Test the creation of a scalable discipline."""

    def f_1(x_1, x_2, x_3):
        return sin(2 * np_pi * x_1) + cos(2 * np_pi * x_2) + x_3

    def f_2(x_1, x_2, x_3):
        return sin(2 * np_pi * x_1) * cos(2 * np_pi * x_2) - x_3

    data = IODataset(dataset_name="sinus")
    x1_val = x2_val = x3_val = linspace(0.0, 1.0, 10)[:, newaxis]
    data.add_variable("x1", x1_val, data.INPUT_GROUP)
    data.add_variable("x2", x2_val, data.INPUT_GROUP)
    data.add_variable("x3", x2_val, data.INPUT_GROUP)
    data.add_variable("y1", f_1(x1_val, x2_val, x3_val), data.OUTPUT_GROUP)
    data.add_variable("y2", f_2(x1_val, x2_val, x3_val), data.OUTPUT_GROUP)
    create_scalable("ScalableDiagonalModel", data, fill_factor=0.7)


def test_create_mda() -> None:
    """Test the creation of an MDA from the Sobieski disciplines."""
    disciplines = create_discipline([
        "SobieskiAerodynamics",
        "SobieskiPropulsion",
        "SobieskiStructure",
        "SobieskiMission",
    ])
    mda = create_mda("MDAGaussSeidel", disciplines)
    mda.execute()
    assert mda.residual_history[-1] < 1e-4


def test_get_available_mdas() -> None:
    """Test that the available MDA solvers are retrieved correctly."""
    mdas = get_available_mdas()
    assert "MDAGaussSeidel" in mdas
    assert "MDA" not in mdas


def test_get_discipline_inputs_schema() -> None:
    """Test that the discipline input schemas are retrieved correctly."""
    mission = create_discipline("SobieskiMission")
    schema_dict = get_discipline_inputs_schema(mission)
    for key in mission.io.input_grammar:
        assert key in schema_dict["properties"]

    schema_str = get_discipline_inputs_schema(mission, True)
    assert isinstance(schema_str, str)
    get_discipline_inputs_schema(mission, pretty_print=True)


def test_get_discipline_outputs_schema() -> None:
    """Test that the discipline output schemas are retrieved correctly."""
    mission = create_discipline("SobieskiMission")
    schema_dict = get_discipline_outputs_schema(mission)
    for key in mission.io.output_grammar:
        assert key in schema_dict["properties"]

    schema_str = get_discipline_outputs_schema(mission, True)
    assert isinstance(schema_str, str)
    get_discipline_outputs_schema(mission, pretty_print=True)


def test_get_scenario_differentiation_modes() -> None:
    """Test that the scenario differentiation modes are retrieved correctly."""
    modes = get_scenario_differentiation_modes()
    for mode in modes:
        assert isinstance(mode, str)


def test_get_formulation_options_schema() -> None:
    """Test that the formulation options schemas are retrieved correctly."""
    mdf_schema = get_formulation_options_schema("MDF")
    for prop in ["differentiated_input_names_substitute", "main_mda_settings"]:
        assert prop in mdf_schema["properties"]

    idf_schema = get_formulation_options_schema("IDF")
    for prop in [
        "differentiated_input_names_substitute",
        "normalize_constraints",
        "n_processes",
        "use_threading",
    ]:
        assert prop in idf_schema["properties"]
    get_formulation_options_schema("IDF", pretty_print=True)


def test_get_discipline_options_schema() -> None:
    """Test that the discipline options schemas are retrieved correctly."""
    for disc in ["SobieskiMission", "MDOChain", "AnalyticDiscipline"]:
        schema = get_discipline_options_schema(disc)
        props = schema["properties"]

        for opts in [
            "jac_approx_type",
            "linearization_mode",
            "cache_hdf_file",
            "cache_hdf_node_path",
        ]:
            assert opts in props
        get_discipline_options_schema(disc, pretty_print=True)


def test_get_discipline_options_defaults() -> None:
    """Test that the discipline options defaults are retrieved correctly."""
    for disc in ["SobieskiMission", "MDOChain", "AnalyticDiscipline"]:
        defaults = get_discipline_options_defaults(disc)
        assert len(defaults) > 0


def test_get_default_sub_option_values() -> None:
    """Test that the default sub option values are retrieved correctly."""
    defaults = get_formulations_sub_options_defaults("MDF", main_mda_name="MDAChain")
    assert defaults is not None

    defaults = get_formulations_sub_options_defaults("DisciplinaryOpt")
    assert defaults == {}


def test_get_formulations_options_defaults() -> None:
    """Test that the formulation options defaults are retrieved correctly."""
    for form in ["MDF", "BiLevel"]:
        defaults = get_formulations_options_defaults(form)
        assert len(defaults) > 0


def test_get_available_scenario_types() -> None:
    """Test that the available scenario types are retrieved correctly."""
    scen_types = get_available_scenario_types()
    assert "MDO" in scen_types
    assert "DOE" in scen_types


def test_create_parameter_space() -> None:
    """Test the creation of a parameter space."""
    parameter_space = create_parameter_space()
    parameter_space.add_variable(
        "name", type_="float", lower_bound=-1, upper_bound=1, value=0
    )
    parameter_space.add_random_variable("other_name", "OTNormalDistribution")
    parameter_space.check()


def test_create_design_space() -> None:
    """Test the creation of a design space."""
    design_space = create_design_space()
    design_space.add_variable(
        "name", type_="float", lower_bound=-1, upper_bound=1, value=0
    )
    design_space.check()


def test_write_design_space(tmp_wd) -> None:
    """Test that a design space can be exported to a text or h5 file.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    design_space = create_design_space()
    design_space.add_variable(
        "name", type_="float", lower_bound=-1, upper_bound=1, value=0
    )
    write_design_space(design_space, "design_space.csv")
    write_design_space(design_space, "design_space.h5")


def test_create_cache() -> None:
    """Test the creation of a cache."""
    cache = create_cache("MemoryFullCache")
    assert not cache


def test_get_available_caches() -> None:
    """Test that the available caches are retrieved correctly."""
    caches = get_available_caches()
    # plugins may add classes
    assert set(caches) <= {"HDF5Cache", "MemoryFullCache", "SimpleCache"}


@pytest.mark.parametrize(
    ("dataset_name", "expected_n_samples"),
    [("BurgersDataset", 30), ("IrisDataset", 150), ("RosenbrockDataset", 100)],
)
def test_create_benchmark_dataset(tmp_wd, dataset_name, expected_n_samples) -> None:
    """Test the load_dataset method with the `BurgersDataset`.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        dataset_name: The dataset to consider.
        expected_n_samples: The expected number of samples.
    """
    dataset = create_benchmark_dataset(dataset_name)
    assert len(dataset) == expected_n_samples


def test_print_configuration(capfd) -> None:
    """Check that the |g| configuration is shown to the user.

    Args:
        capfd: Fixture capture outputs sent to `stdout` and
            `stderr`.
    """
    print_configuration()

    out, err = capfd.readouterr()
    assert not err

    expected = """Settings
   Discipline
      The caches are enabled.
      The counters are enabled.
      The input data are checked before running the discipline.
      The output data are checked after running the discipline.
   ProblemFunction
      The counters are enabled.
   BaseDriverLibrary
      The progress bar is enabled."""

    assert expected in out

    gemseo_modules = [
        "BaseOptimizationLibrary",
        "BaseDOELibrary",
        "BaseRegressor",
        "BaseMDOFormulation",
        "BaseMDA",
        "BasePost",
    ]

    for module in gemseo_modules:
        header_patterns = (
            r"\+-+\+$\n"
            rf"\|\s+{module}\s+\|$\n"
            r"\+-+\+-+\+-+\+$\n"
            r"\|\s+Module\s+\|\s+Is available\?\s+\|\s+Purpose or error "
            r"message\s+\|$\n"
        )

        expected = re.compile(header_patterns, re.MULTILINE)
        assert bool(re.search(expected, out))


@pytest.fixture(scope="module")
def variables_space():
    """A mock design space."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=2.0, value=1.0)
    design_space.add_variable("y", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    return design_space


@pytest.mark.parametrize(
    "settings",
    [{"n_samples": 4}, {"settings_model": PYDOE_FULLFACT_Settings(n_samples=4)}],
)
@pytest.mark.parametrize(
    ("transformation", "expected_points"),
    [
        ({"unit_sampling": True}, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ({}, [[0.0, -1.0], [2.0, -1.0], [0.0, 1.0], [2.0, 1.0]]),
    ],
)
def test_compute_doe(
    variables_space, settings, transformation, expected_points
) -> None:
    """Check the computation of a DOE in a variables space."""
    points = compute_doe(
        variables_space, algo_name="PYDOE_FULLFACT", **settings, **transformation
    )
    assert (points == array(expected_points)).all()


def test_import_analytic_discipline(tmp_wd) -> None:
    """Check that an analytic discipline performs correctly after import."""
    file_path = "saved_discipline.pkl"

    discipline = create_discipline("AnalyticDiscipline", expressions={"y": "2*x"})
    to_pickle(discipline, file_path)
    discipline.execute()

    loaded_discipline = import_discipline(file_path, AnalyticDiscipline)
    loaded_discipline.execute()

    assert loaded_discipline.io.data["x"] == discipline.io.data["x"]
    assert loaded_discipline.io.data["y"] == discipline.io.data["y"]


def test_import_discipline(tmp_wd) -> None:
    """Check that a discipline performs correctly after import."""
    file_path = "saved_discipline.pkl"

    discipline = create_discipline("Sellar1")
    to_pickle(discipline, file_path)
    discipline.execute()

    loaded_discipline = import_discipline(file_path)
    loaded_discipline.execute()

    assert loaded_discipline.io.data["x_1"] == discipline.io.data["x_1"]
    assert loaded_discipline.io.data["y_1"] == discipline.io.data["y_1"]


@pytest.mark.parametrize("enable_discipline_statistics", [False, True])
@pytest.mark.parametrize("enable_function_statistics", [False, True])
@pytest.mark.parametrize("enable_progress_bar", [False, True])
@pytest.mark.parametrize("enable_discipline_cache", [False, True])
@pytest.mark.parametrize("validate_input_data", [False, True])
@pytest.mark.parametrize("validate_output_data", [False, True])
def test_configure(
    enable_discipline_statistics,
    enable_function_statistics,
    enable_progress_bar,
    enable_discipline_cache,
    validate_input_data,
    validate_output_data,
) -> None:
    """Check that the configuration of GEMSEO works correctly."""
    configure(
        enable_discipline_statistics=enable_discipline_statistics,
        enable_function_statistics=enable_function_statistics,
        enable_progress_bar=enable_progress_bar,
        enable_discipline_cache=enable_discipline_cache,
        validate_input_data=validate_input_data,
        validate_output_data=validate_output_data,
    )
    assert ProblemFunction.enable_statistics == enable_function_statistics
    assert ExecutionStatistics.is_enabled == enable_discipline_statistics
    assert Discipline.validate_input_data == validate_input_data
    assert Discipline.validate_output_data == validate_output_data
    assert Discipline.default_cache_type == (
        Discipline.CacheType.SIMPLE
        if enable_discipline_cache
        else Discipline.CacheType.NONE
    )
    assert BaseDriverLibrary.enable_progress_bar == enable_progress_bar
    assert BaseMDA.default_cache_type == Discipline.CacheType.SIMPLE
    configure()


def test_configure_default() -> None:
    """Check the default use of configure."""
    configure()
    assert ProblemFunction.enable_statistics is True
    assert ExecutionStatistics.is_enabled is True
    assert Discipline.validate_input_data is True
    assert Discipline.validate_output_data is True
    assert Discipline.default_cache_type == Discipline.CacheType.SIMPLE
    assert BaseDriverLibrary.enable_progress_bar is True


def test_wrap_discipline_in_job_scheduler(tmpdir) -> None:
    """Test the job scheduler API."""
    disc = create_discipline("SobieskiMission")
    wrapped = wrap_discipline_in_job_scheduler(
        disc,
        scheduler_name="LSF",
        workdir_path=tmpdir,
        scheduler_run_command="python",
        job_template_path=Path(__file__).parent
        / "disciplines"
        / "wrappers"
        / "job_schedulers"
        / "mock_job_scheduler.py",
        job_out_filename="run_disc.py",
    )
    assert "y_4" in wrapped.execute()


def test_create_dataset_without_name() -> None:
    """Check create_dataset without name."""
    assert create_dataset().name == "Dataset"
    assert create_dataset(class_name=DatasetClassName.IODataset).name == "IODataset"


def test_create_dataset_class_name() -> None:
    """Check create_dataset with class_name set from the enum DatasetClassName."""
    isinstance(create_dataset(class_name=DatasetClassName.IODataset), IODataset)


def test_configure_logger() -> None:
    """Check configure_logger() with default argument values."""
    logger = configure_logger()
    assert logger == logging.root
    assert logger.level == logging.INFO


def test_configure_logger_name() -> None:
    """Check configure_logger() with custom name."""
    logger = configure_logger(logger_name="foo")
    assert logger.name == "foo"


def test_configure_logger_level() -> None:
    """Check configure_logger() with custom level."""
    logger = configure_logger(level=logging.WARNING)
    assert logger.level == logging.WARNING


def test_configure_logger_format(caplog) -> None:
    """Check configure_logger() with custom message and date formats."""
    date_format = "foo"
    message_format = "%(levelname)8s / %(asctime)s: %(message)s"
    logger = configure_logger(
        logger_name="bar", date_format=date_format, message_format=message_format
    )
    logger.info("baz")
    assert LOGGING_SETTINGS.date_format == date_format
    assert LOGGING_SETTINGS.message_format == message_format
    assert re.match(r"INFO     bar:test_gemseo\.py:\d+\d+\d+ baz\n", caplog.text)


def test_configure_logger_file(tmp_wd) -> None:
    """Check configure_logger() with custom file."""
    logger = configure_logger(filename="foo.txt")
    stream_handler = logger.handlers[0]
    assert isinstance(stream_handler, MultiLineStreamHandler)
    assert len(logger.handlers) == 2
    file_handler = logger.handlers[-1]
    assert Path(file_handler.baseFilename) == tmp_wd / "foo.txt"
    assert file_handler.mode == "a"
    assert file_handler.delay
    assert file_handler.encoding == "utf-8"
    assert file_handler.formatter == stream_handler.formatter


def test_configure_logger_file_mode(tmp_wd) -> None:
    """Check configure_logger() with custom file and file mode."""
    logger = configure_logger(filename="foo.txt", filemode="w")
    assert logger.handlers[-1].mode == "w"


@pytest.fixture(scope="module")
def disciplines() -> list[AnalyticDiscipline]:
    """Two simple disciplines to be sampled, with 1 input and 2 outputs."""
    return [
        AnalyticDiscipline({"out1": "2*inpt"}),
        AnalyticDiscipline({"out2": "3*inpt+out1"}),
    ]


@pytest.fixture(scope="module")
def input_space() -> DesignSpace:
    """The input space on which to sample the discipline."""
    design_space = DesignSpace()
    design_space.add_variable("inpt", lower_bound=1.0, upper_bound=2.0)
    return design_space


@pytest.mark.parametrize(
    "output_names",
    ["out1", ["out1", "out2"]],
)
def test_sample_disciplines(disciplines, input_space, output_names, caplog):
    """Check the sampling of two disciplines."""
    dataset = sample_disciplines(
        disciplines, input_space, output_names, algo_name="PYDOE_FULLFACT", n_samples=2
    )
    assert dataset.name == "Sampling"

    assert_equal(
        dataset.get_view(variable_names="inpt").to_numpy(), array([[1.0], [2.0]])
    )

    if isinstance(output_names, str):
        output_names = [output_names]

    assert_equal(
        dataset.get_view(variable_names="out1").to_numpy(), array([[2.0], [4.0]])
    )

    if "out2" in output_names:
        assert_equal(
            dataset.get_view(variable_names="out2").to_numpy(), array([[5.0], [10.0]])
        )
    else:
        assert "out2" not in dataset.variable_names

    # By default, the gradients are not sampled.
    assert "@out1" not in dataset.variable_names

    assert "Objective" not in caplog.text


def test_sample_disciplines_options(disciplines, input_space, caplog):
    """Check the sampling of two disciplines with options."""
    dataset = sample_disciplines(
        disciplines,
        input_space,
        "out1",
        algo_name="PYDOE_FULLFACT",
        n_samples=2,
        name="foo",
        # Use DisciplinaryOpt instead of MDF
        formulation_name="DisciplinaryOpt",
        # Sample -objective instead of objective
        formulation_settings={"maximize_objective": True},
        # Sample the gradients
        eval_jac=True,
        # Log the problem
        log_problem=True,
    )
    assert dataset.name == "foo"
    assert_equal(
        dataset.get_view(variable_names="-out1").to_numpy(), array([[-2.0], [-4.0]])
    )
    assert_equal(
        dataset.get_view(variable_names="@-out1").to_numpy(), array([[-2.0], [-2.0]])
    )
    assert "Objective" in caplog.text


def test_sample_disciplines_backup_file(disciplines, input_space, tmp_wd):
    """Check that sample_disciplines can use a backup file."""
    with mock.patch.object(DOEScenario, "set_optimization_history_backup") as method:
        sample_disciplines(
            disciplines,
            input_space,
            ["out1", "out2"],
            algo_name="PYDOE_FULLFACT",
            n_samples=2,
        )

    method.assert_not_called()

    sample_disciplines(
        disciplines,
        input_space,
        ["out1", "out2"],
        algo_name="PYDOE_FULLFACT",
        backup_settings=BackupSettings("database.hdf5"),
        n_samples=2,
    )
    assert len(Database.from_hdf("database.hdf5")) == 2

    with mock.patch.object(DOEScenario, "set_optimization_history_backup") as method:
        sample_disciplines(
            disciplines,
            input_space,
            ["out1", "out2"],
            algo_name="PYDOE_FULLFACT",
            backup_settings=BackupSettings("database.hdf5"),
            n_samples=2,
        )

    assert method.call_args.kwargs == {
        "at_each_iteration": False,
        "at_each_function_call": True,
        "erase": False,
        "load": False,
    }

    with mock.patch.object(DOEScenario, "set_optimization_history_backup") as method:
        sample_disciplines(
            disciplines,
            input_space,
            ["out1", "out2"],
            algo_name="PYDOE_FULLFACT",
            backup_settings=BackupSettings(
                "database.hdf5",
                at_each_iteration=True,
                at_each_function_call=False,
                erase=True,
                load=True,
            ),
            n_samples=2,
        )

    assert method.call_args.kwargs == {
        "at_each_iteration": True,
        "at_each_function_call": False,
        "erase": True,
        "load": True,
    }


def test_generate_xdsm():
    """Verify that generate_xdsm works correctly."""
    with mock.patch.object(XDSMizer, "run") as run:
        generate_xdsm(Sellar1())

    assert run.call_args.kwargs == {
        "directory_path": ".",
        "file_name": "xdsm",
        "pdf_batchmode": True,
        "pdf_build": True,
        "pdf_cleanup": True,
        "save_html": True,
        "save_json": False,
        "save_pdf": False,
        "show_html": False,
    }

    with mock.patch.object(XDSMizer, "run") as run:
        generate_xdsm(
            Sellar1(),
            directory_path="a",
            file_name="b",
            show_html="c",
            save_html="d",
            save_json="e",
            save_pdf="f",
            pdf_build="g",
            pdf_cleanup="h",
            pdf_batchmode="i",
        )

    assert run.call_args.kwargs == {
        "directory_path": "a",
        "file_name": "b",
        "pdf_batchmode": "i",
        "pdf_build": "g",
        "pdf_cleanup": "h",
        "save_html": "d",
        "save_json": "e",
        "save_pdf": "f",
        "show_html": "c",
    }


def test_generate_xdsm_return():
    """Verify that generate_xdsm returns an XDSM."""
    assert isinstance(generate_xdsm(Sellar1(), save_html=False), XDSM)


@pytest.mark.parametrize(
    ("file_path", "names", "size"),
    [
        ("database_from_database_default.hdf5", ["input"], 2),
        ("database_from_database_custom.hdf5", ["a", "b"], 1),
        ("database_from_problem.hdf5", ["x"], 2),
    ],
)
def test_import_database(file_path, names, size, caplog):
    """Check import_database."""
    database = import_database(Path(__file__).parent / file_path)
    assert isinstance(database, Database)
    input_space = database.input_space
    assert database.input_space.variable_names == names
    for name in names:
        assert input_space.get_size(name) == size

    assert ("Importing the database" in caplog.text) is ("from_database" in file_path)
