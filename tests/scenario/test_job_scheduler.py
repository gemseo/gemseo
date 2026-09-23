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
"""Tests for wrapping a scenario in a job scheduler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import isfinite
from numpy import square

from gemseo.core.function.array_function import ArrayFunction
from gemseo.discipline.wrapper.job_scheduler.lsf import LSF
from gemseo.discipline.wrapper.job_scheduler.slurm import SLURM
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.formulation.idf_settings import IDF_Settings
from gemseo.formulation.mdf_settings import MDF_Settings
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.mdo.sellar.sellar_1 import Sellar1
from gemseo.problem.mdo.sellar.sellar_2 import Sellar2
from gemseo.problem.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.problem.mdo.sellar.sellar_system import SellarSystem
from gemseo.scenario.adapter.evaluation import EvaluationScenarioAdapter
from gemseo.scenario.adapter.mdo import MDOScenarioAdapter
from gemseo.scenario.evaluation import EvaluationScenario
from gemseo.scenario.job_scheduler import wrap_scenario_in_job_scheduler
from gemseo.scenario.mdo import MDOScenario
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.discipline.wrapper.job_scheduler.discipline import (
        JobSchedulerDiscipline,
    )

MOCK_TEMPLATE_PATH = (
    Path(__file__).parent.parent
    / "discipline"
    / "wrapper"
    / "job_scheduler"
    / "mock_job_scheduler.py"
)


@pytest.fixture
def create_scenario() -> Callable[[type[EvaluationScenario]], EvaluationScenario]:
    """A function creating a scenario for the Sellar problem."""

    def create(
        scenario_class: type[EvaluationScenario], use_idf: bool = False
    ) -> EvaluationScenario:
        """Create the scenario.

        Args:
            scenario_class: The class of the scenario.
            use_idf: Whether to use the IDF formulation instead of MDF.

        Returns:
            The scenario, without objective, constraint, observable nor algorithm.
        """
        return scenario_class(
            [Sellar1(), Sellar2(), SellarSystem()],
            SellarDesignSpace(add_couplings=use_idf),
            formulation_settings=IDF_Settings() if use_idf else MDF_Settings(),
        )

    return create


@pytest.fixture
def create_sellar_scenario(create_scenario) -> Callable[[], MDOScenario]:
    """A function creating a Sellar MDO scenario with MDF formulation."""

    def create() -> MDOScenario:
        """Create the scenario.

        Returns:
            The configured scenario (without algorithm set).
        """
        scenario = create_scenario(MDOScenario)
        scenario.add_objective("obj")
        scenario.add_constraint("c_1", constraint_type="ineq")
        scenario.add_observable("c_2")
        return scenario

    return create


@pytest.fixture
def create_sellar_scenario_with_standardized_functions(
    create_scenario,
) -> Callable[[], MDOScenario]:
    """A function creating a Sellar MDO scenario with standardized function names.

    The objective is maximized and the constraint has a non-zero value,
    so that the standardized function names
    (``-obj`` and ``[c_1-1.0]``)
    are not the names of the discipline outputs.
    """

    def create() -> MDOScenario:
        """Create the scenario.

        Returns:
            The configured scenario (without algorithm set).
        """
        scenario = create_scenario(MDOScenario)
        scenario.add_objective("obj", minimize=False)
        scenario.add_constraint("c_1", constraint_type="ineq", value=1.0)
        return scenario

    return create


@pytest.fixture
def create_evaluation_scenario(create_scenario) -> Callable[[], EvaluationScenario]:
    """A function creating a Sellar evaluation scenario with MDF formulation."""

    def create() -> EvaluationScenario:
        """Create the scenario.

        Returns:
            The configured scenario (without algorithm set).
        """
        scenario = create_scenario(EvaluationScenario)
        scenario.add_observable("obj")
        scenario.add_observable("c_1")
        scenario.add_observable("c_2")
        return scenario

    return create


@pytest.fixture
def create_doe_settings() -> Callable[[], CustomDOE_Settings]:
    """A function creating deterministic DOE settings for the Sellar design space."""

    def create() -> CustomDOE_Settings:
        """Create the DOE settings.

        Returns:
            The settings of a custom DOE with two samples.
        """
        return CustomDOE_Settings(
            samples=array([[0.8, 0.9, 3.5, 2.5], [1.2, 1.1, 4.5, 3.5]])
        )

    return create


@pytest.fixture
def wrap_in_mocked_scheduler(
    tmp_wd,
) -> Callable[[EvaluationScenario], JobSchedulerDiscipline]:
    """A function wrapping a scenario in a mocked SLURM job scheduler."""

    def wrap(scenario: EvaluationScenario) -> JobSchedulerDiscipline:
        """Wrap a scenario in a mocked SLURM job scheduler.

        Args:
            scenario: The scenario to be wrapped.

        Returns:
            The wrapped scenario.
        """
        return wrap_scenario_in_job_scheduler(
            scenario,
            "SLURM",
            workdir_path=tmp_wd,
            job_template_path=MOCK_TEMPLATE_PATH,
            job_out_filename="run_disc.py",
            scheduler_run_command="python",
        )

    return wrap


@pytest.mark.parametrize(
    ("scheduler_name", "scheduler_class"), [("SLURM", SLURM), ("LSF", LSF)]
)
def test_scheduler_name(
    tmp_wd, create_sellar_scenario, scheduler_name, scheduler_class
) -> None:
    """Test that the wrapper class depends on the name of the job scheduler."""
    scenario = create_sellar_scenario()
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))
    wrapper = wrap_scenario_in_job_scheduler(
        scenario, scheduler_name, workdir_path=tmp_wd
    )
    assert isinstance(wrapper, scheduler_class)


def test_sellar_scenario_remote_execution(
    create_sellar_scenario, wrap_in_mocked_scheduler
) -> None:
    """Test execution of a Sellar scenario via a mocked job scheduler."""
    scenario = create_sellar_scenario()
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=20))

    # Compare with a local reference execution.
    ref_scenario = create_sellar_scenario()
    ref_scenario.execute(algorithm_settings=SLSQP_Settings(max_iter=20))
    ref_obj = ref_scenario.formulation.problem.solution.f_opt

    wrapper = wrap_in_mocked_scheduler(scenario)
    wrapper.execute()

    assert isfinite(wrapper.io.output_data["obj"]).all()

    assert isfinite(ref_obj)
    assert abs(wrapper.io.output_data["obj"][0] - ref_obj) < 1e-3


DESIGN_VARIABLE_NAMES = {"x_1", "x_2", "x_shared"}


@pytest.mark.parametrize(
    ("create", "use_doe", "input_names", "output_names"),
    [
        ("create_sellar_scenario", False, DESIGN_VARIABLE_NAMES, {"obj", "c_1", "c_2"}),
        ("create_sellar_scenario", True, set(), {"obj", "c_1", "c_2"}),
        (
            "create_sellar_scenario_with_standardized_functions",
            False,
            DESIGN_VARIABLE_NAMES,
            {"obj", "c_1"},
        ),
        ("create_evaluation_scenario", True, set(), {"obj", "c_1", "c_2"}),
    ],
)
def test_wrapper_grammars(
    request,
    create,
    use_doe,
    input_names,
    output_names,
    create_doe_settings,
    wrap_in_mocked_scheduler,
) -> None:
    """Test that the wrapper grammars are derived from the scenario.

    The wrapper of a scenario driven by a DOE algorithm has no input.
    """
    scenario = request.getfixturevalue(create)()
    scenario.set_algorithm(
        algorithm_settings=create_doe_settings()
        if use_doe
        else SLSQP_Settings(max_iter=1)
    )
    wrapper = wrap_in_mocked_scheduler(scenario)
    assert set(wrapper.io.input_grammar) == input_names
    assert set(wrapper.io.output_grammar) == output_names | DESIGN_VARIABLE_NAMES


def test_wrapper_grammars_with_several_top_level_disciplines(
    create_scenario, wrap_in_mocked_scheduler
) -> None:
    """Test the wrapper grammars when the formulation has several top-level disciplines.

    With IDF,
    the outputs of the functions are spread over the three top-level disciplines.
    """
    scenario = create_scenario(MDOScenario, use_idf=True)
    scenario.add_objective("obj")
    scenario.add_constraint("c_1", constraint_type="ineq")
    scenario.add_observable("c_2")
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))
    assert len(scenario.formulation.get_top_level_disciplines()) == 3
    wrapper = wrap_in_mocked_scheduler(scenario)
    design_variable_names = DESIGN_VARIABLE_NAMES | {"y_1", "y_2"}
    assert set(wrapper.io.input_grammar) == design_variable_names
    assert set(wrapper.io.output_grammar) == {"obj", "c_1", "c_2"} | (
        design_variable_names
    )


def test_doe_ignores_starting_point(
    create_sellar_scenario, create_doe_settings, wrap_in_mocked_scheduler
) -> None:
    """Test that the wrapper of a scenario driven by a DOE does not set x0."""
    scenario = create_sellar_scenario()
    scenario.set_algorithm(algorithm_settings=create_doe_settings())
    wrapper = wrap_in_mocked_scheduler(scenario)
    assert wrapper._discipline._set_x0_before_exec is False


def test_design_variables_as_starting_point(
    create_sellar_scenario, wrap_in_mocked_scheduler
) -> None:
    """Test that the input design variables are the starting point of the scenario."""
    scenario = create_sellar_scenario()
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))
    x_shared = array([3.0, 3.0])

    ref_scenario = create_sellar_scenario()
    current_value = ref_scenario.design_space.get_current_value(as_dict=True)
    ref_scenario.design_space.set_current_value({**current_value, "x_shared": x_shared})
    ref_scenario.execute(algorithm_settings=SLSQP_Settings(max_iter=1))
    ref_obj = ref_scenario.optimization_result.f_opt

    wrapper = wrap_in_mocked_scheduler(scenario)
    wrapper.execute({"x_shared": x_shared})

    assert wrapper.io.output_data["x_shared"] == pytest.approx(x_shared)
    assert wrapper.io.output_data["obj"] == pytest.approx(ref_obj)


def test_design_variable_defaults(
    create_sellar_scenario, wrap_in_mocked_scheduler
) -> None:
    """Initialize the wrapper defaults from the current design space."""
    scenario = create_sellar_scenario()
    scenario.design_space.set_current_value(array([1.2, 1.1, 4.5, 3.5]))
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))

    wrapper = wrap_in_mocked_scheduler(scenario)

    for name, value in scenario.design_space.get_current_value(as_dict=True).items():
        assert wrapper.io.input_grammar.defaults[name] == pytest.approx(value)


def test_current_design_value_as_starting_point(
    create_sellar_scenario, wrap_in_mocked_scheduler
) -> None:
    """Preserve the configured starting point when executing without inputs."""
    scenario = create_sellar_scenario()
    scenario.design_space.set_current_value(array([1.2, 1.1, 4.5, 3.5]))
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))
    current_value = scenario.design_space.get_current_value(as_dict=True)

    wrapper = wrap_in_mocked_scheduler(scenario)
    wrapper.execute()

    for name, value in current_value.items():
        assert wrapper.io.output_data[name] == pytest.approx(value)


def test_adapter_settings(tmp_wd, create_sellar_scenario) -> None:
    """Test that the adapter settings are passed to the scenario adapter."""
    scenario = create_sellar_scenario()
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=2))
    wrapper = wrap_scenario_in_job_scheduler(
        scenario,
        "SLURM",
        workdir_path=tmp_wd,
        adapter_settings={
            "output_names": ["obj"],
            "save_databases": True,
            "database_file_prefix": "history",
        },
        job_template_path=MOCK_TEMPLATE_PATH,
        job_out_filename="run_disc.py",
        scheduler_run_command="python",
    )
    assert set(wrapper.io.output_grammar) == {"obj"}
    wrapper.execute()
    assert list(Path(tmp_wd).glob("*/history_*.h5"))


@pytest.mark.parametrize(
    ("adapter_settings", "set_x0_before_exec"),
    [
        ({}, True),
        ({"reset_x0_before_exec": True}, False),
        ({"input_names": ["x_shared"]}, True),
        ({"input_names": ["x_shared"], "set_x0_before_exec": True}, True),
        ({"input_names": ["x_shared"], "set_x0_before_exec": False}, False),
    ],
)
def test_default_set_x0_before_exec(
    tmp_wd, create_sellar_scenario, adapter_settings, set_x0_before_exec
) -> None:
    """Test the default starting point setting of the scenario adapter."""
    scenario = create_sellar_scenario()
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))
    wrapper = wrap_scenario_in_job_scheduler(
        scenario, "SLURM", workdir_path=tmp_wd, adapter_settings=adapter_settings
    )
    assert wrapper._discipline._set_x0_before_exec is set_x0_before_exec


@pytest.mark.parametrize("set_x0_before_exec", [None, True, False])
def test_partial_design_variable_inputs(
    tmp_wd, create_sellar_scenario, set_x0_before_exec
) -> None:
    """Test a wrapper whose inputs are only a part of the design variables.

    The other design variables keep their current value as starting point.
    When the starting point is not set from the inputs, the latter are ignored.
    """
    scenario = create_sellar_scenario()
    scenario.design_space.set_current_value(array([1.2, 1.1, 4.5, 3.5]))
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))
    adapter_settings = {"input_names": ["x_shared"]}
    if set_x0_before_exec is not None:
        adapter_settings["set_x0_before_exec"] = set_x0_before_exec

    current_value = scenario.design_space.get_current_value(as_dict=True)
    x_shared = array([3.0, 3.0])
    if set_x0_before_exec is not False:
        current_value["x_shared"] = x_shared
    ref_scenario = create_sellar_scenario()
    ref_scenario.design_space.set_current_value(current_value)
    ref_scenario.execute(algorithm_settings=SLSQP_Settings(max_iter=1))

    wrapper = wrap_scenario_in_job_scheduler(
        scenario,
        "SLURM",
        workdir_path=tmp_wd,
        adapter_settings=adapter_settings,
        job_template_path=MOCK_TEMPLATE_PATH,
        job_out_filename="run_disc.py",
        scheduler_run_command="python",
    )
    assert set(wrapper.io.input_grammar) == {"x_shared"}
    wrapper.execute({"x_shared": x_shared})
    for name, value in current_value.items():
        assert wrapper.io.output_data[name] == pytest.approx(value)
    assert wrapper.io.output_data["obj"] == pytest.approx(
        ref_scenario.optimization_result.f_opt
    )


@pytest.mark.parametrize(
    ("create", "adapter_class"),
    [
        ("create_sellar_scenario", MDOScenarioAdapter),
        ("create_evaluation_scenario", EvaluationScenarioAdapter),
    ],
)
def test_adapter_class(
    request, create, adapter_class, create_doe_settings, wrap_in_mocked_scheduler
) -> None:
    """Test that the adapter class depends on the problem of the scenario."""
    scenario = request.getfixturevalue(create)()
    scenario.set_algorithm(algorithm_settings=create_doe_settings())
    wrapper = wrap_in_mocked_scheduler(scenario)
    assert type(wrapper._discipline) is adapter_class


@pytest.mark.parametrize(
    "create", ["create_sellar_scenario", "create_evaluation_scenario"]
)
def test_wrap_scenario_requires_algorithm(
    request, create, wrap_in_mocked_scheduler, snapshot
) -> None:
    """Test that wrapping a scenario without a set algorithm raises an error."""
    scenario = request.getfixturevalue(create)()
    with assert_exception(ValueError, snapshot):
        wrap_in_mocked_scheduler(scenario)


def test_wrap_scenario_requires_objective(
    create_scenario, wrap_in_mocked_scheduler, snapshot
) -> None:
    """Test that wrapping an MDO scenario without an objective raises an error."""
    scenario = create_scenario(MDOScenario)
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))
    with assert_exception(ValueError, snapshot):
        wrap_in_mocked_scheduler(scenario)


def test_evaluation_scenario_remote_execution(
    create_evaluation_scenario, create_doe_settings, wrap_in_mocked_scheduler
) -> None:
    """Test execution of a Sellar evaluation scenario via a mocked job scheduler."""
    scenario = create_evaluation_scenario()
    scenario.set_algorithm(algorithm_settings=create_doe_settings())

    # Compare with a local reference execution.
    ref_scenario = create_evaluation_scenario()
    ref_scenario.execute(algorithm_settings=create_doe_settings())
    ref_database = ref_scenario.formulation.problem.database

    wrapper = wrap_in_mocked_scheduler(scenario)
    wrapper.execute()

    for name in ("obj", "c_1", "c_2"):
        assert name in wrapper.io.output_data
        assert isfinite(wrapper.io.output_data[name]).all()

    # The adapter returns the outputs of the last DOE sample.
    last_x = ref_database.get_x_vect(len(ref_database))
    ref_output_data = ref_database.get_function_value("obj", last_x)
    assert wrapper.io.output_data["obj"][0] == pytest.approx(
        float(ref_output_data), rel=1e-6
    )


@pytest.mark.parametrize("output_name", ["x_square", "alpha"])
def test_function_outputs_missing_from_disciplines(
    create_sellar_scenario, wrap_in_mocked_scheduler, caplog, output_name
) -> None:
    """Test that the function outputs the disciplines cannot produce are skipped.

    `alpha` is an input of the top-level discipline but an output of none.
    """
    scenario = create_sellar_scenario()
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))
    # The observable must be picklable to be sent to the remote process.
    scenario.formulation.problem.add_observable(
        ArrayFunction(square, "observable", output_names=[output_name])
    )

    wrapper = wrap_in_mocked_scheduler(scenario)

    assert set(wrapper.io.output_grammar) == {
        "obj",
        "c_1",
        "c_2",
        "x_1",
        "x_2",
        "x_shared",
    }
    assert (
        f"The function outputs '{output_name}' of the scenario {scenario.name} "
        "are not outputs of its top-level disciplines "
        "and cannot be returned by the job scheduler wrapper."
    ) in caplog.text
    wrapper.execute()
    assert isfinite(wrapper.io.output_data["obj"]).all()


def test_function_outputs_missing_from_disciplines_with_output_names(
    tmp_wd, create_sellar_scenario, caplog
) -> None:
    """Test that no output is reported as skipped when output_names is passed."""
    scenario = create_sellar_scenario()
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=1))
    scenario.formulation.problem.add_observable(
        ArrayFunction(square, "x_square", output_names=["x_square"])
    )
    wrapper = wrap_scenario_in_job_scheduler(
        scenario,
        "SLURM",
        workdir_path=tmp_wd,
        adapter_settings={"output_names": ["obj"]},
    )
    assert set(wrapper.io.output_grammar) == {"obj"}
    assert "cannot be returned by the job scheduler wrapper" not in caplog.text


def test_scenario_with_standardized_function_names(
    create_sellar_scenario_with_standardized_functions, wrap_in_mocked_scheduler
) -> None:
    """Test wrapping a scenario whose function names are not the output names."""
    scenario = create_sellar_scenario_with_standardized_functions()
    scenario.set_algorithm(algorithm_settings=SLSQP_Settings(max_iter=10))

    # Compare with a local reference execution.
    ref_scenario = create_sellar_scenario_with_standardized_functions()
    ref_scenario.execute(algorithm_settings=SLSQP_Settings(max_iter=10))
    ref_result = ref_scenario.optimization_result

    wrapper = wrap_in_mocked_scheduler(scenario)
    wrapper.execute()

    # The wrapper returns the discipline outputs, not the standardized values.
    output_data = wrapper.io.output_data
    assert output_data["obj"] == pytest.approx(-ref_result.f_opt)
    assert output_data["c_1"] == pytest.approx(
        ref_result.constraint_values["[c_1-1.0]"] + 1.0
    )
    x_opt = ref_scenario.design_space.convert_array_to_dict(ref_result.x_opt)
    for name, value in x_opt.items():
        assert output_data[name] == pytest.approx(value)
