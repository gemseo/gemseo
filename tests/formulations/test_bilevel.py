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

import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from gemseo import create_discipline
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import NLOPT_COBYLA_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.core.chains.warm_started_chain import WarmStartedDisciplineChain
from gemseo.core.discipline import Discipline
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.formulations.bilevel import BiLevel
from gemseo.formulations.bilevel_bcd import BiLevelBCD
from gemseo.formulations.bilevel_settings import BiLevel_Settings
from gemseo.formulations.disciplinary_opt_settings import DisciplinaryOpt_Settings
from gemseo.formulations.factory import MDO_FORMULATION_FACTORY
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo import MDOScenario
from gemseo.utils.discipline import get_sub_disciplines
from gemseo.utils.name_generator import NameGenerator
from gemseo.utils.testing.bilevel_test_helper import create_aerostructure_scenario
from gemseo.utils.testing.bilevel_test_helper import create_dummy_bilevel_scenario
from gemseo.utils.testing.bilevel_test_helper import (
    create_sobieski_bilevel_bcd_scenario,
)
from gemseo.utils.testing.bilevel_test_helper import create_sobieski_bilevel_scenario
from gemseo.utils.testing.bilevel_test_helper import create_sobieski_sub_scenarios
from gemseo.utils.testing.disciplines_creator import create_disciplines_from_desc

from ..mda.mda_gauss_seidel import SobieskiMDAGaussSeidel

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture(params=["BiLevel", "BiLevelBCD"])
def formulation_name(request):
    """The name of a bi-level formulation."""
    return request.param


@pytest.fixture
def generate_sobieski_bilevel_scenario() -> Callable[..., MDOScenario]:
    """Generate a BiLevel scenario for the Sobieski's SSBJ problem."""
    return create_sobieski_bilevel_scenario()


@pytest.fixture
def generate_sobieski_bilevel_bcd_scenario() -> Callable[..., MDOScenario]:
    """Generate a BiLevelBCD scenario for the Sobieski's SSBJ problem."""
    return create_sobieski_bilevel_bcd_scenario()


@pytest.fixture
def dummy_bilevel_scenario(formulation_name) -> MDOScenario:
    """Fixture from an existing function."""
    return create_dummy_bilevel_scenario(formulation_name)


@pytest.fixture
def aerostructure_scenario(formulation_name) -> MDOScenario:
    return create_aerostructure_scenario(formulation_name)


@pytest.fixture
def sobieski_sub_scenarios() -> tuple[MDOScenario, MDOScenario, MDOScenario]:
    """Fixture from an existing function."""
    return create_sobieski_sub_scenarios()


def test_constraint_not_in_sub_scenario(generate_sobieski_bilevel_scenario) -> None:
    """Test the execution of the Sobieski BiLevel Scenario."""
    scenario = generate_sobieski_bilevel_scenario(apply_constraints_to_system=False)

    for i in range(1, 4):
        scenario.add_constraint(f"g_{i}", constraint_type=scenario.ConstraintType.INEQ)

    for i in range(3):
        cstrs = scenario.disciplines[i].formulation.problem.constraints
        assert len(cstrs) == 1
        assert cstrs[0].name == f"g_{i + 1}"

    cstrs_sys = scenario.formulation.problem.constraints
    assert len(cstrs_sys) == 0
    with pytest.raises(ValueError):
        scenario.add_constraint("toto", constraint_type=scenario.ConstraintType.INEQ)


def test_get_sub_options_grammar_errors() -> None:
    """Test that errors are raised if no MDA name is provided."""
    with pytest.raises(ValueError):
        BiLevel.get_sub_options_grammar()
    with pytest.raises(ValueError):
        BiLevel.get_default_sub_option_values()


def test_get_sub_options_grammar() -> None:
    """Test that the MDAJacobi sub-options can be retrieved."""
    sub_options_schema = BiLevel.get_sub_options_grammar(main_mda_name="MDAJacobi")
    assert sub_options_schema.name == "MDAJacobi"

    sub_option_values = BiLevel.get_default_sub_option_values(main_mda_name="MDAJacobi")
    assert "acceleration_method" in sub_option_values


def test_bilevel_aerostructure(aerostructure_scenario) -> None:
    """Test the Bi-level formulation on the aero-structure problem."""
    scenario = aerostructure_scenario

    assert isinstance(scenario.optimization_result, OptimizationResult)
    assert scenario.formulation.problem.database.n_iterations == 5


def test_bilevel_weak_couplings(dummy_bilevel_scenario) -> None:
    """Test that the adapters contains the discipline weak couplings.

    This test generates a bi-level scenario which does not aim to be run as it has no
    physical significance. It is checked that the weak couplings are present in the
    inputs (resp. outputs) of the adapters, if they are in the top_level inputs (resp.
    outputs) of the adapter.
    """
    # a and b are weak couplings of all the disciplines,
    # and they are in the top-level outputs of the first adapter
    disciplines = dummy_bilevel_scenario.formulation.chain.disciplines
    assert "a" in disciplines[0].io.output_grammar
    assert "b" in disciplines[1].io.output_grammar

    if dummy_bilevel_scenario.formulation == BiLevel:
        assert "b" in disciplines[0].io.input_grammar
        assert "a" in disciplines[1].io.input_grammar

    if dummy_bilevel_scenario.formulation == BiLevelBCD:
        assert "b" in disciplines[0].io.output_grammar
        assert "a" in disciplines[1].io.output_grammar
        assert "a" not in disciplines[0].io.input_grammar
        assert "b" not in disciplines[0].io.input_grammar
        assert "a" not in disciplines[1].io.input_grammar
        assert "b" not in disciplines[1].io.input_grammar


@pytest.mark.parametrize("formulation_name", ["BiLevel"], indirect=True)
def test_bilevel_mda_getter(dummy_bilevel_scenario) -> None:
    """Test that the user can access the MDA1 and MDA2."""
    # In the Dummy scenario, there's not strongly coupled disciplines -> No MDA1
    assert dummy_bilevel_scenario.formulation.mda1 is None
    assert "obj" in dummy_bilevel_scenario.formulation.mda2.io.output_grammar


@pytest.mark.parametrize("formulation_name", ["BiLevel"], indirect=True)
def test_bilevel_mda_setter(dummy_bilevel_scenario) -> None:
    """Test that the user cannot modify the MDA1 and MDA2 after instantiation."""
    discipline = create_discipline("SellarSystem")
    with pytest.raises(AttributeError):
        dummy_bilevel_scenario.formulation.mda1 = discipline
    with pytest.raises(AttributeError):
        dummy_bilevel_scenario.formulation.mda2 = discipline


@pytest.mark.parametrize(
    "scenario",
    [generate_sobieski_bilevel_scenario, generate_sobieski_bilevel_bcd_scenario],
)
def test_get_sub_disciplines(scenario, request) -> None:
    """Test the get_sub_disciplines method with the BiLevel formulation.

    Args:
        scenario: Fixture to instantiate a Sobieski BiLevel Scenario.
    """
    scenario = request.getfixturevalue(scenario.__name__)()
    classes = [
        discipline.__class__
        for discipline in get_sub_disciplines(scenario.formulation.disciplines)
    ]

    assert set(classes) == {
        SobieskiPropulsion().__class__,
        SobieskiMission().__class__,
        SobieskiAerodynamics().__class__,
        SobieskiStructure().__class__,
    }


@pytest.mark.parametrize(
    "scenario",
    [generate_sobieski_bilevel_scenario, generate_sobieski_bilevel_bcd_scenario],
)
def test_bilevel_warm_start(scenario, request) -> None:
    """Test the warm start of the BiLevel chain.

    Args:
        scenario: Fixture to instantiate a Sobieski BiLevel Scenario.
    """
    scenario = request.getfixturevalue(scenario.__name__)()
    scenario.formulation.chain.set_cache(Discipline.CacheType.MEMORY_FULL)
    bilevel_chain_cache = scenario.formulation.chain.cache
    scenario.formulation.chain.disciplines[0].set_cache(
        Discipline.CacheType.MEMORY_FULL
    )
    mda1_cache = scenario.formulation.chain.disciplines[0].cache
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=3))
    mda1_inputs = [entry.inputs for entry in mda1_cache.get_all_entries()]
    chain_outputs = [entry.outputs for entry in bilevel_chain_cache.get_all_entries()]

    assert mda1_inputs[1]["y_21"] == chain_outputs[0]["y_21"]
    assert (mda1_inputs[1]["y_12"] == chain_outputs[0]["y_12"]).all()
    assert mda1_inputs[2]["y_21"] == chain_outputs[1]["y_21"]
    assert (mda1_inputs[2]["y_12"] == chain_outputs[1]["y_12"]).all()
    assert mda1_inputs[1]["y_32"] == chain_outputs[0]["y_32"]
    assert (mda1_inputs[1]["y_23"] == chain_outputs[0]["y_23"]).all()
    assert mda1_inputs[2]["y_32"] == chain_outputs[1]["y_32"]
    assert (mda1_inputs[2]["y_23"] == chain_outputs[1]["y_23"]).all()


@pytest.mark.parametrize("formulation_name", ["BiLevel"], indirect=True)
def test_bilevel_warm_start_no_mda1(dummy_bilevel_scenario) -> None:
    """Test that a warm start chain is built even if the process does not include any
    MDA1.
    """
    assert isinstance(
        dummy_bilevel_scenario.formulation.chain, WarmStartedDisciplineChain
    )


@pytest.mark.parametrize("formulation_name", ["BiLevel"], indirect=True)
def test_bilevel_get_variable_names_to_warm_start_without_mdas(
    dummy_bilevel_scenario, monkeypatch
) -> None:
    """ " Check that the warm start properly considers the adapter variables when there
    are no MDAs."""

    def _no_mda2(*args, **kwargs):
        return None

    monkeypatch.setattr(dummy_bilevel_scenario.formulation, "_mda2", _no_mda2())
    variables = []
    for adapter in dummy_bilevel_scenario.formulation._scenario_adapters:
        variables.extend(adapter.io.output_grammar)
    assert sorted(set(variables)) == sorted(
        dummy_bilevel_scenario.formulation._get_variable_names_to_warm_start()
    )


def test_bilevel_get_variable_names_to_warm_start_from_mdas(
    generate_sobieski_bilevel_scenario,
) -> None:
    """Check that the variables from both MDAs are being considered in the warm
    start."""
    scenario = generate_sobieski_bilevel_scenario()
    for variable in scenario.formulation._mda1.io.output_grammar:
        assert variable in scenario.formulation._get_variable_names_to_warm_start()
    for variable in scenario.formulation._mda2.io.output_grammar:
        assert variable in scenario.formulation._get_variable_names_to_warm_start()


@pytest.mark.parametrize(
    ("settings", "sub_scenario_formulation", "scenario_formulation"),
    [
        ({}, "MDF", "BiLevelBCD"),
        ({"sub_scenarios_log_level": None}, "MDF", "BiLevelBCD"),
        ({"sub_scenarios_log_level": logging.INFO}, "MDF", "BiLevelBCD"),
        ({"sub_scenarios_log_level": logging.WARNING}, "MDF", "BiLevelBCD"),
        ({}, "DisciplinaryOpt", "BiLevel"),
        ({"sub_scenarios_log_level": None}, "DisciplinaryOpt", "BiLevel"),
        ({"sub_scenarios_log_level": logging.INFO}, "DisciplinaryOpt", "BiLevel"),
        ({"sub_scenarios_log_level": logging.WARNING}, "DisciplinaryOpt", "BiLevel"),
    ],
)
def test_scenario_log_level(
    caplog, settings, sub_scenario_formulation, scenario_formulation
) -> None:
    """Check scenario_log_level."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=0.5)
    sub_scenario = MDOScenario(
        [AnalyticDiscipline({"z": "(x+y)**2"})],
        design_space.filter(["y"], copy=True),
        name="FooScenario",
        formulation_settings=MDO_FORMULATION_FACTORY.get_class(
            sub_scenario_formulation
        ).settings_class(),
    )
    sub_scenario.add_objective("z")
    sub_scenario.set_algorithm(NLOPT_COBYLA_Settings(max_iter=2))
    scenario = MDOScenario(
        [sub_scenario],
        design_space.filter(["x"]),
        formulation_settings=MDO_FORMULATION_FACTORY.get_class(
            scenario_formulation
        ).settings_class(**settings),
    )
    scenario.add_objective("z")
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=2))
    sub_scenarios_log_level = settings.get("sub_scenarios_log_level")
    if sub_scenarios_log_level == logging.WARNING:
        assert "Start FooScenario execution" not in caplog.text
    else:
        assert "Start FooScenario execution" in caplog.text


def test_remove_couplings_from_ds(sobieski_sub_scenarios, caplog) -> None:
    """Check the removal of strong couplings for the design space."""
    problem = OptimizationProblem(SobieskiProblem().design_space)
    formulation = BiLevel(problem, [*sobieski_sub_scenarios, SobieskiMission()])
    problem.objective = formulation.create_objective(["y_4"])
    for strong_coupling in ["y_12", "y_21", "y_23", "y_31", "y_32"]:
        assert strong_coupling not in formulation.design_space
    assert (
        "The coupling variable y_12 was removed from the design space." in caplog.text
    )


@pytest.mark.parametrize(
    ("scenario", "subscenario"),
    [
        (
            generate_sobieski_bilevel_bcd_scenario,
            {
                "StructureScenario_adapter": {
                    "ssbj_local_variables": {"x_1", "x_2", "x_3"},
                    "ssbj_input_couplings": {"y_31", "y_21", "y_23"},
                },
                "PropulsionScenario_adapter": {
                    "ssbj_local_variables": {"x_1", "x_2", "x_3"},
                    "ssbj_input_couplings": {"y_31", "y_21", "y_23"},
                },
                "AerodynamicsScenario_adapter": {
                    "ssbj_local_variables": {"x_1", "x_2", "x_3"},
                    "ssbj_input_couplings": {"y_31", "y_21", "y_23"},
                },
            },
        ),
        (
            generate_sobieski_bilevel_scenario,
            {
                "StructureScenario_adapter": {
                    "ssbj_local_variables": {"x_1"},
                    "ssbj_input_couplings": {"y_31", "y_21"},
                },
                "PropulsionScenario_adapter": {
                    "ssbj_local_variables": {"x_3"},
                    "ssbj_input_couplings": {"y_23"},
                },
                "AerodynamicsScenario_adapter": {
                    "ssbj_local_variables": {"x_2"},
                    "ssbj_input_couplings": {"y_32", "y_12"},
                },
            },
        ),
    ],
)
def test_adapters_inputs_outputs(scenario, subscenario, request) -> None:
    """Test that the ScenarioAdapters within the BCD loop have the right inputs and
    outputs.
    """
    scenario = request.getfixturevalue(scenario.__name__)()
    all_ssbj_couplings = {
        "y_14",
        "y_12",
        "y_32",
        "y_31",
        "y_34",
        "y_21",
        "y_23",
        "y_24",
    }

    # Necessary couplings as inputs,
    # depends on the order of the disciplines within the block MDAs.
    ssbj_shared_variables = {"x_shared"}
    for scenario_adapter in scenario.formulation._scenario_adapters:
        ssbj_local_variables = subscenario[scenario_adapter.name][
            "ssbj_local_variables"
        ]
        ssbj_input_couplings = subscenario[scenario_adapter.name][
            "ssbj_input_couplings"
        ]
        adapter = scenario_adapter

        design_variable = set(
            adapter.scenario.formulation.problem.design_space.variable_names
        )
        other_local = ssbj_local_variables.difference(design_variable)
        # Check the inputs
        inputs = set(adapter.io.input_grammar)
        # Check the outputs
        outputs = set(adapter.io.output_grammar)
        # Shared variables should always be present.
        assert ssbj_shared_variables.issubset(inputs)
        # Only necessary couplings should always be present.
        assert ssbj_input_couplings.issubset(inputs)
        # All local variables, excepted the optimized ones, should be present.
        assert other_local.issubset(inputs)
        assert not design_variable.issubset(inputs)

        # Shared variables should never be present
        assert not ssbj_shared_variables.issubset(outputs)
        # Only the optimized local variables should be present.
        assert design_variable.issubset(outputs)

        if isinstance(scenario.formulation, BiLevelBCD):
            # All couplings should always be present
            assert all_ssbj_couplings.issubset(outputs)
            assert not other_local.issubset(outputs)


@pytest.mark.parametrize(
    ("sub_scenario_formulation", "scenario_formulation"),
    [
        ("MDF", "BiLevelBCD"),
        ("MDF", "BiLevel"),
    ],
)
def test_system_variables_not_in_variables_to_warm_start(
    sub_scenario_formulation, scenario_formulation
):
    """Test that the system variables are not in the list of variables to warm start.

    This test simulates a very particular configuration in which one of the design
    variables of the system-level scenario is included in the MDA1 of the BiLevel Chain.

    The BiLevel formulation uses a warm-start mechanism for all variables that are
    outputs of either the MDA1, the MDA2, or the sub-scenarios. Here, we verify that
    even in this particular configuration, the system variable is not included in the
    list of variables to warm-start.

    This is important because if it were included, the value provided by the solver for
    the design variable would be replaced with the value from the previous iteration.
    And this would occur at the _execute level, so the system level optimizer would not
    "see" the replaced value.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("b", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("baz", lower_bound=0.0, upper_bound=1.0, value=0.5)
    sub_scenario_1 = MDOScenario(
        [AnalyticDiscipline({"z": "(x+y)**2", "b": "c+y"}, "foo")],
        design_space.filter(["y"], copy=True),
        formulation_settings=MDO_FORMULATION_FACTORY.get_class(
            sub_scenario_formulation
        ).settings_class(),
        name="FooScenario",
    )
    sub_scenario_1.add_objective("z")
    sub_scenario_1.set_algorithm(NLOPT_COBYLA_Settings(max_iter=2))

    sub_scenario_2 = MDOScenario(
        [AnalyticDiscipline({"c": "(x+b)**2"}, "bar")],
        design_space.filter(["b"], copy=True),
        name="BarScenario",
        formulation_settings=MDO_FORMULATION_FACTORY.get_class(
            sub_scenario_formulation
        ).settings_class(),
    )
    sub_scenario_2.add_objective("c")
    sub_scenario_2.set_algorithm(NLOPT_COBYLA_Settings(max_iter=2))

    scenario = MDOScenario(
        [
            sub_scenario_1,
            sub_scenario_2,
            AnalyticDiscipline({"qux": "baz"}),
            AnalyticDiscipline({"x": "x"}),
        ],
        design_space.filter(["x", "baz"]),
        formulation_settings=MDO_FORMULATION_FACTORY.get_class(
            scenario_formulation
        ).settings_class(apply_constraints_to_sub_scenarios=False),
    )
    scenario.add_objective("z")
    assert "x" not in scenario.formulation.chain._variable_names_to_warm_start
    assert "baz" not in scenario.formulation.chain._variable_names_to_warm_start


def test_bcd_mda(generate_sobieski_bilevel_bcd_scenario):
    """Test that a BCD MDA is created and included in the chain."""
    scenario = generate_sobieski_bilevel_bcd_scenario()
    assert scenario.formulation.bcd_mda
    assert scenario.formulation.bcd_mda == scenario.formulation.chain.disciplines[1]


@pytest.mark.parametrize("keep_opt_history", [True, False])
def test_keep_opt_history(generate_sobieski_bilevel_scenario, keep_opt_history) -> None:
    """Test the keep_opt_history setting."""
    scenario = generate_sobieski_bilevel_scenario(keep_opt_history=keep_opt_history)
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=2))
    assert len(scenario.formulation.scenario_adapters[0].databases) == (
        2 if keep_opt_history else 0
    )


@pytest.mark.parametrize(
    ("save_opt_history", "naming"),
    [
        (True, NameGenerator.Naming.NUMBERED),
        (True, NameGenerator.Naming.UUID),
        (False, NameGenerator.Naming.NUMBERED),
        (False, NameGenerator.Naming.UUID),
    ],
)
def test_save_opt_history(
    tmp_wd, generate_sobieski_bilevel_scenario, save_opt_history, naming
) -> None:
    """Test the save_opt_history and naming settings."""
    scenario = generate_sobieski_bilevel_scenario(
        save_opt_history=save_opt_history, naming=naming
    )
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=2))
    # path_structure= Path("StructureScenario")
    # path_aero = Path("AerodynamicsScenario")
    path_propulsion = Path("PropulsionScenario")
    if naming == NameGenerator.Naming.NUMBERED:
        assert (
            path_propulsion.parent / f"{path_propulsion.name}_1.h5"
        ).exists() is save_opt_history
        assert (
            path_propulsion.parent / f"{path_propulsion.name}_2.h5"
        ).exists() is save_opt_history
    else:
        assert (
            len(list(tmp_wd.rglob(f"{path_propulsion.name}_*.h5"))) == 2
        ) is save_opt_history


@pytest.mark.parametrize(
    "scenario",
    [generate_sobieski_bilevel_scenario, generate_sobieski_bilevel_bcd_scenario],
)
@pytest.mark.parametrize("include_sub_formulations", [False, True])
def test_get_top_level_disciplines(scenario, request, include_sub_formulations) -> None:
    """Test the get_top_level_disciplines method."""
    scenario = request.getfixturevalue(scenario.__name__)()
    bilevel = scenario.formulation
    top_level_disciplines = bilevel.get_top_level_disciplines(
        include_sub_formulations=include_sub_formulations
    )
    if include_sub_formulations:
        adapters = bilevel.scenario_adapters
        assert top_level_disciplines == (
            bilevel.chain,
            adapters[0].scenario.formulation.get_top_level_disciplines()[0],
            adapters[1].scenario.formulation.get_top_level_disciplines()[0],
            adapters[2].scenario.formulation.get_top_level_disciplines()[0],
        )
    else:
        assert top_level_disciplines == (bilevel.chain,)


@pytest.mark.parametrize(
    "main_mda",
    [
        {"main_mda_settings": MDAGaussSeidel_Settings()},
        {"main_mda_settings": MDAGaussSeidel_Settings(max_mda_iter=10)},
    ],
)
def test_main_mda_settings(main_mda):
    """Tests that BiLevel supports main_mda_settings as dict and Pydantic model."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=0.5)
    sub_scenario = MDOScenario(
        [AnalyticDiscipline({"z": "(x+y)**2"})],
        design_space.filter(["y"], copy=True),
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    sub_scenario.add_objective("z")
    sub_scenario.set_algorithm(NLOPT_COBYLA_Settings(max_iter=2))
    scenario = MDOScenario(
        [sub_scenario],
        design_space.filter(["x"]),
        formulation_settings=BiLevel_Settings(**main_mda),
    )
    scenario.add_objective("z")
    assert isinstance(scenario.formulation.mda2, MDAGaussSeidel)


def test_bilevel_settings_error():
    """Check the error when reset_x0_before_opt and set_x0_before_opt are both True."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The options reset_x0_before_opt and set_x0_before_opt of BiLevel "
            "cannot both be True"
        ),
    ):
        BiLevel_Settings(reset_x0_before_opt=True, set_x0_before_opt=True)


@pytest.mark.parametrize(
    "kwargs", [{"set_x0_before_opt": False}, {"set_x0_before_opt": True}, {}]
)
def test_set_x0_before_opt(generate_sobieski_bilevel_scenario, kwargs):
    """Verify that set_x0_before_opt is passed to MDOScenarioAdapter"""
    set_x0_before_opt = kwargs.get("set_x0_before_opt", False)
    scenario = generate_sobieski_bilevel_scenario(**kwargs)
    scenario_adapters = scenario.formulation._scenario_adapters
    assert scenario_adapters[0]._set_x0_before_opt is set_x0_before_opt
    assert scenario_adapters[1]._set_x0_before_opt is set_x0_before_opt
    assert scenario_adapters[2]._set_x0_before_opt is set_x0_before_opt


def test_optimal_local_design_history(generate_sobieski_bilevel_scenario):
    """Test the database contains the optimal values of the local design variables."""
    scenario = generate_sobieski_bilevel_scenario()
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=1))
    last_item = scenario.formulation.problem.database.last_item
    assert set(last_item) == {"x_3", "x_1", "-y_4", "x_2"}
    y_4 = last_item["-y_4"]
    scenario.execute(NLOPT_COBYLA_Settings(max_iter=2))
    last_item = scenario.formulation.problem.database.last_item
    assert last_item["-y_4"] != y_4


@pytest.mark.parametrize("apply_constraints_to_system", [None, False, True])
@pytest.mark.parametrize("apply_constraints_to_sub_scenarios", [None, False, True])
@pytest.mark.parametrize("apply_to_system_level", [None, False, True])
@pytest.mark.parametrize("apply_to_sub_level", [None, False, True])
def test_constraint_level_policy(
    generate_sobieski_bilevel_scenario,
    apply_constraints_to_system,
    apply_constraints_to_sub_scenarios,
    apply_to_system_level,
    apply_to_sub_level,
):
    settings = {}
    if apply_constraints_to_system is not None:
        settings["apply_constraints_to_system"] = apply_constraints_to_system
    if apply_constraints_to_sub_scenarios is not None:
        settings["apply_constraints_to_sub_scenarios"] = (
            apply_constraints_to_sub_scenarios
        )

    scenario = generate_sobieski_bilevel_scenario(**settings)

    kwargs = {}
    if apply_to_system_level is not None:
        kwargs["apply_to_system_level"] = apply_to_system_level
    if apply_to_sub_level is not None:
        kwargs["apply_to_sub_level"] = apply_to_sub_level

    scenario.add_constraint("g_1", **kwargs)

    if apply_constraints_to_system is None:
        apply_constraints_to_system = True
    if apply_constraints_to_sub_scenarios is None:
        apply_constraints_to_sub_scenarios = True
    assert bool(scenario.formulation.problem.constraints) is (
        apply_constraints_to_system
        if apply_to_system_level is None
        else apply_to_system_level
    )
    sub_scenario = scenario.formulation.disciplines[0]
    assert bool(sub_scenario.formulation.problem.constraints) is (
        apply_constraints_to_sub_scenarios
        if apply_to_sub_level is None
        else apply_to_sub_level
    )


def test_execute_custom_mdas(generate_sobieski_bilevel_scenario, caplog):
    """Test the bilevel execution when MDA instances are provided directly.

    Here, we test that a scenario with custom MDAs gives the same optimal value than
    a scenario with automatically built MDAs."""
    scenario = generate_sobieski_bilevel_scenario(
        apply_constraints_to_sub_scenarios=True,
        apply_constraints_to_system=False,
        use_mda1=False,
    )

    for i in range(1, 4):
        scenario.add_constraint(["g_" + str(i)], ArrayFunction.ConstraintType.INEQ)

    scenario.execute(NLOPT_COBYLA_Settings(max_iter=1))

    mda1 = SobieskiMDAGaussSeidel()
    mda2 = SobieskiMDAGaussSeidel()

    scenario2 = generate_sobieski_bilevel_scenario(
        apply_constraints_to_sub_scenarios=True,
        apply_constraints_to_system=False,
        mda1_instance=mda1,
        mda2_instance=mda2,
    )

    for i in range(1, 4):
        scenario2.add_constraint(["g_" + str(i)], ArrayFunction.ConstraintType.INEQ)

    scenario2.execute(NLOPT_COBYLA_Settings(max_iter=1))

    assert (
        scenario.optimization_result.x_opt == scenario2.optimization_result.x_opt
    ).all()
    assert "Using the provided MDA1 instance" in caplog.text
    assert "Using the provided MDA2 instance" in caplog.text


def test_bilevel_no_mda2():
    """Test that the MDA2 can be deactivated in the bi-level formulation.

    In this BiLevel formulation, the user has the possibility to inject a custom MDA1
    (and even an object inheriting from Discipline). The MDA2 is deactivated, and it is
    checked that the adapters output their top-level output variables. This test checks
    that, as well as the impacts on the grammars of the adapters.
    """
    disc_expressions = {
        "disc_1": (["x_1"], ["a"]),
        "disc_2": (["a", "x_1", "x_2"], ["b", "obj"]),
    }
    discipline_1, discipline_2 = create_disciplines_from_desc(disc_expressions)

    system_design_space = DesignSpace()
    system_design_space.add_variable("x_1")

    sub_design_space_1 = DesignSpace()
    sub_design_space_1.add_variable("x_2")
    sub_scenario_1 = MDOScenario(
        [discipline_2],
        sub_design_space_1,
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    sub_scenario_1.add_objective("obj")

    scenario = MDOScenario(
        [sub_scenario_1],
        system_design_space,
        formulation_settings=BiLevel_Settings(
            use_mda1=True,
            use_mda2=False,
            mda1_instance=discipline_1,
        ),
    )
    scenario.add_objective("obj")

    assert "a" in scenario.formulation.chain.disciplines[0].io.output_grammar.names
    assert "a" in scenario.formulation.chain.disciplines[1].io.input_grammar.names

    # use_mda2 is set to False -> all the top_level outputs are in the adapter outputs
    assert "obj" in scenario.formulation.chain.disciplines[1].io.output_grammar.names
    assert "b" in scenario.formulation.chain.disciplines[1].io.output_grammar.names


def test_bilevel_no_mda1(generate_sobieski_bilevel_scenario, caplog):
    """Test that the MDA1 can be deactivated in the bi-level formulation."""
    scenario = generate_sobieski_bilevel_scenario(
        apply_constraints_to_sub_scenarios=True,
        apply_constraints_to_system=False,
        use_mda1=False,
    )
    assert scenario.formulation.mda1 is None
    assert (
        "The first MDA has been deactivated in the Bilevel formulation" in caplog.text
    )


def test_bilevel_custom_mda1_and_custom_mda2():
    """Test that custom MDAs can be defined in the bi-level formulation.

    In this BiLevel formulation, the user has the possibility to inject custom MDAs
    (and even an object inheriting from MDODiscipline). This test has not physical
    significance, and only checks the correct set-up of the grammar adapter.
    """
    disc_expressions = {
        "disc_1": (["x_1"], ["a"]),
        "disc_2": (["a", "x_2"], ["b"]),
        "disc_3": (["a", "b", "x_1", "x_2"], ["obj"]),
    }

    discipline_1, discipline_2, discipline_3 = create_disciplines_from_desc(
        disc_expressions
    )

    system_design_space = DesignSpace()
    system_design_space.add_variable("x_1")

    sub_design_space_1 = DesignSpace()
    sub_design_space_1.add_variable("x_2")
    sub_scenario_1 = MDOScenario(
        [discipline_2, discipline_3],
        sub_design_space_1,
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    sub_scenario_1.add_objective("obj")

    scenario = MDOScenario(
        [sub_scenario_1],
        system_design_space,
        formulation_settings=BiLevel_Settings(
            use_mda1=True,
            use_mda2=True,
            mda1_instance=discipline_1,
            mda2_instance=discipline_3,
        ),
    )
    scenario.add_objective("obj")

    assert "a" in scenario.formulation.chain.disciplines[0].io.output_grammar.names
    assert "a" in scenario.formulation.chain.disciplines[1].io.input_grammar.names

    assert "b" in scenario.formulation.chain.disciplines[1].io.output_grammar.names
    assert "b" in scenario.formulation.chain.disciplines[2].io.input_grammar.names

    assert "obj" in scenario.formulation.chain.disciplines[2].io.output_grammar.names


def test_bilevel_custom_mda1_with_mda2():
    """Test that a custom MDA1 can be defined in the bi-level formulation.

    In this BiLevel formulation, the user has the possibility to inject a custom MDA
    (and even an object inheriting from MDODiscipline). This test has not physical
    significance, and only checks the correct set-up of the grammar adapter.
    """
    disc_expression = {
        "discipline_1": (["x_1"], ["a"]),
        "discipline_2": (["a", "x_2"], ["b"]),
        "discipline_3": (["a", "x_1", "x_2", "b"], ["obj"]),
    }

    discipline_1, discipline_2, discipline_3 = create_disciplines_from_desc(
        disc_expression
    )

    system_design_space = DesignSpace()
    system_design_space.add_variable("x_1")

    sub_design_space_1 = DesignSpace()
    sub_design_space_1.add_variable("x_2")
    sub_scenario_1 = MDOScenario(
        [discipline_2, discipline_3],
        sub_design_space_1,
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    sub_scenario_1.add_objective("obj")

    scenario = MDOScenario(
        [sub_scenario_1],
        system_design_space,
        formulation_settings=BiLevel_Settings(
            use_mda1=True,
            use_mda2=True,
            mda1_instance=discipline_1,
        ),
    )
    scenario.add_objective("obj")

    assert "a" in scenario.formulation.chain.disciplines[0].io.output_grammar.names
    assert "a" in scenario.formulation.chain.disciplines[1].io.input_grammar.names
    assert "b" in scenario.formulation.chain.disciplines[1].io.output_grammar.names
    assert "obj" in scenario.formulation.chain.disciplines[2].io.output_grammar.names


def test_bilevel_warm_start_disciplines_as_subscenario():
    """Test the warm start of the BiLevel chain for disciplines as sub scenarios."""
    ds = SobieskiProblem().design_space
    aerodynamics = SobieskiAerodynamics()
    struct = SobieskiStructure()
    mission = SobieskiMission()

    sc_str = MDOScenario(
        [struct],
        deepcopy(ds).filter("x_1"),
        name="StructureScenario",
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    sc_str.add_objective("y_11", minimize=False)

    # Discipline to be used as sub scenario
    # THIS DISCIPLINE HAS NO REAL MEANING, IT ONLY MOCKS OUTPUTS AS A FUNCTION OF INPUTS
    aero_disc_scenario = AnalyticDiscipline({
        "y_23": "25000.*y_32",
        "y_21": "50000.*y_32",
        "y_24": "4.5*x_2",
        "x_2": "y_32",
    })

    sub_scenarios = [sc_str]
    sub_disciplines = [*sub_scenarios, mission, aerodynamics]
    for sc in sub_scenarios:
        sc.set_algorithm(SLSQP_Settings(max_iter=5))

    ds = SobieskiProblem().design_space
    sc_system = MDOScenario(
        sub_disciplines,
        ds.filter(["x_shared", "y_14"]),
        formulation_settings=BiLevel_Settings(
            disciplines_as_sub_scenario=[aero_disc_scenario],
        ),
    )
    sc_system.add_objective("y_4", minimize=False)

    sc_system.formulation.chain.set_cache(Discipline.CacheType.MEMORY_FULL)
    bilevel_chain_cache = sc_system.formulation.chain.cache
    sc_system.formulation.chain.disciplines[0].set_cache(
        Discipline.CacheType.MEMORY_FULL
    )
    mda1_cache = sc_system.formulation.chain.disciplines[0].cache

    sc_system.formulation._disciplines_as_sub_scenario[0].set_cache(
        Discipline.CacheType.MEMORY_FULL
    )
    disciplines_as_sub_scenario_cache = (
        sc_system.formulation._disciplines_as_sub_scenario[0].cache
    )

    sc_system.execute(NLOPT_COBYLA_Settings(max_iter=3))
    mda1_inputs = [entry.inputs for entry in mda1_cache]
    chain_outputs = [entry.outputs for entry in bilevel_chain_cache]
    disciplines_as_sub_scenario_inputs = [
        entry.inputs for entry in disciplines_as_sub_scenario_cache
    ]

    assert disciplines_as_sub_scenario_inputs[1]["x_2"] == chain_outputs[0]["x_2"]
    assert mda1_inputs[1]["y_21"] == chain_outputs[0]["y_21"]
    assert (mda1_inputs[1]["y_12"] == chain_outputs[0]["y_12"]).all()
    assert mda1_inputs[2]["y_21"] == chain_outputs[1]["y_21"]
    assert (mda1_inputs[2]["y_12"] == chain_outputs[1]["y_12"]).all()


def mock_aero_scenario(x_shared, y_12, y_32):
    """A function that mocks an aerodynamic scenario.

    The computation below makes NO physical sense. It is intended to test the storage
    of the local variables of disciplines as sub-scenarios.
    """
    x_2 = y_32 * 2 + x_shared[0] + y_12[0]
    return x_2  # noqa: RET504


def test_optimal_local_design_history_disciplines_as_sub_scenario():
    """Test the database contains the optimal values of the local design variables.

    In particular, we test the case in which there are disciplines used as
    sub-scenarios.
    """
    structure, _, propulsion = create_sobieski_sub_scenarios()
    structure.set_algorithm(SLSQP_Settings(max_iter=5))
    propulsion.set_algorithm(SLSQP_Settings(max_iter=5))
    aerodynamic_sub_scenario = AutoPyDiscipline(py_func=mock_aero_scenario)

    system_scenario = MDOScenario(
        [structure, propulsion, SobieskiAerodynamics(), SobieskiMission()],
        SobieskiProblem().design_space.filter(["x_shared", "y_14"]),
        formulation_settings=BiLevel_Settings(
            disciplines_as_sub_scenario=[aerodynamic_sub_scenario],
        ),
    )
    system_scenario.add_objective("y_4", minimize=False)

    system_scenario.execute(NLOPT_COBYLA_Settings(max_iter=1))
    last_item = system_scenario.formulation.problem.database.last_item
    assert set(last_item) == {"x_3", "x_1", "-y_4", "x_2"}
    x_2 = last_item["x_2"]
    system_scenario.execute(NLOPT_COBYLA_Settings(max_iter=2))
    last_item = system_scenario.formulation.problem.database.last_item
    assert last_item["x_2"] != x_2


def test_custom_mda2_with_mda1(generate_sobieski_bilevel_scenario, caplog):
    """Test the bilevel scenario with a custom MDA2 and a non-custom MDA1."""
    disc_expression = {
        "discipline_1": (["x_1"], ["a"]),
        "discipline_2": (["a", "x_2"], ["b", "x_1"]),
        "discipline_3": (["a", "x_1", "x_2", "b"], ["obj"]),
    }

    discipline_1, discipline_2, discipline_3 = create_disciplines_from_desc(
        disc_expression
    )

    system_design_space = DesignSpace()
    system_design_space.add_variable("x_1")

    sub_design_space_1 = DesignSpace()
    sub_design_space_1.add_variable("x_2")
    sub_scenario_1 = MDOScenario(
        [discipline_1, discipline_2, discipline_3],
        sub_design_space_1,
        formulation_settings=DisciplinaryOpt_Settings(),
    )
    sub_scenario_1.add_objective("obj")

    scenario = MDOScenario(
        [sub_scenario_1],
        system_design_space,
        formulation_settings=BiLevel_Settings(
            use_mda1=True,
            use_mda2=True,
            mda2_instance=discipline_3,
        ),
    )
    scenario.add_objective("obj")

    assert "x_1" in scenario.formulation.chain.disciplines[0].io.input_grammar.names
    assert "x_1" in scenario.formulation.chain.disciplines[1].io.output_grammar.names
    assert "b" in scenario.formulation.chain.disciplines[1].io.output_grammar.names
    assert "obj" in scenario.formulation.chain.disciplines[2].io.output_grammar.names
