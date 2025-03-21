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
from pathlib import Path

import pytest

from gemseo import create_discipline
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.core.chains.warm_started_chain import MDOWarmStartedChain
from gemseo.core.discipline import Discipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.bilevel import BiLevel
from gemseo.formulations.bilevel_bcd import BiLevelBCD
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.discipline import get_sub_disciplines
from gemseo.utils.name_generator import NameGenerator
from gemseo.utils.testing.bilevel_test_helper import create_aerostructure_scenario
from gemseo.utils.testing.bilevel_test_helper import create_dummy_bilevel_scenario
from gemseo.utils.testing.bilevel_test_helper import (
    create_sobieski_bilevel_bcd_scenario,
)
from gemseo.utils.testing.bilevel_test_helper import create_sobieski_bilevel_scenario
from gemseo.utils.testing.bilevel_test_helper import create_sobieski_sub_scenarios


@pytest.fixture(params=["BiLevel", "BiLevelBCD"])
def scenario_formulation_name(request):
    return request.param


@pytest.fixture
def sobieski_bilevel_scenario():
    """Fixture from an existing function."""
    return create_sobieski_bilevel_scenario()


@pytest.fixture
def sobieski_bilevel_bcd_scenario():
    """Fixture from an existing function."""
    return create_sobieski_bilevel_bcd_scenario()


@pytest.fixture
def dummy_bilevel_scenario(scenario_formulation_name) -> MDOScenario:
    """Fixture from an existing function."""
    return create_dummy_bilevel_scenario(scenario_formulation_name)


@pytest.fixture
def aerostructure_scenario(scenario_formulation_name) -> MDOScenario:
    return create_aerostructure_scenario(scenario_formulation_name)


@pytest.fixture
def sobieski_sub_scenarios() -> tuple[MDOScenario, MDOScenario, MDOScenario]:
    """Fixture from an existing function."""
    return create_sobieski_sub_scenarios()


def test_constraint_not_in_sub_scenario(sobieski_bilevel_scenario) -> None:
    """Test the execution of the Sobieski BiLevel Scenario."""
    scenario = sobieski_bilevel_scenario(
        apply_cstr_tosub_scenarios=True, apply_cstr_to_system=False
    )

    for i in range(1, 4):
        scenario.add_constraint(["g_" + str(i)], constraint_type="ineq")

    for i in range(3):
        cstrs = scenario.disciplines[i].formulation.optimization_problem.constraints
        assert len(cstrs) == 1
        assert cstrs[0].name == "g_" + str(i + 1)

    cstrs_sys = scenario.formulation.optimization_problem.constraints
    assert len(cstrs_sys) == 0
    with pytest.raises(ValueError):
        scenario.add_constraint(["toto"], constraint_type="ineq")


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
    assert scenario.formulation.optimization_problem.database.n_iterations == 5


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


@pytest.mark.parametrize("scenario_formulation_name", ["BiLevel"], indirect=True)
def test_bilevel_mda_getter(dummy_bilevel_scenario) -> None:
    """Test that the user can access the MDA1 and MDA2."""
    # In the Dummy scenario, there's not strongly coupled disciplines -> No MDA1
    assert dummy_bilevel_scenario.formulation.mda1 is None
    assert "obj" in dummy_bilevel_scenario.formulation.mda2.io.output_grammar


@pytest.mark.parametrize("scenario_formulation_name", ["BiLevel"], indirect=True)
def test_bilevel_mda_setter(dummy_bilevel_scenario) -> None:
    """Test that the user cannot modify the MDA1 and MDA2 after instantiation."""
    discipline = create_discipline("SellarSystem")
    with pytest.raises(AttributeError):
        dummy_bilevel_scenario.formulation.mda1 = discipline
    with pytest.raises(AttributeError):
        dummy_bilevel_scenario.formulation.mda2 = discipline


@pytest.mark.parametrize(
    "scenario", [sobieski_bilevel_scenario, sobieski_bilevel_bcd_scenario]
)
def test_get_sub_disciplines(scenario, request) -> None:
    """Test the get_sub_disciplines method with the BiLevel formulation.

    Args:
        sobieski_bilevel_scenario: Fixture to instantiate a Sobieski BiLevel Scenario.
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
    "scenario", [sobieski_bilevel_scenario, sobieski_bilevel_bcd_scenario]
)
def test_bilevel_warm_start(scenario, request) -> None:
    """Test the warm start of the BiLevel chain.

    Args:
        sobieski_bilevel_scenario: Fixture to instantiate a Sobieski BiLevel Scenario.
    """
    scenario = request.getfixturevalue(scenario.__name__)()
    scenario.formulation.chain.set_cache(Discipline.CacheType.MEMORY_FULL)
    bilevel_chain_cache = scenario.formulation.chain.cache
    scenario.formulation.chain.disciplines[0].set_cache(
        Discipline.CacheType.MEMORY_FULL
    )
    mda1_cache = scenario.formulation.chain.disciplines[0].cache
    scenario.execute(algo_name="NLOPT_COBYLA", max_iter=3)
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


@pytest.mark.parametrize("scenario_formulation_name", ["BiLevel"], indirect=True)
def test_bilevel_warm_start_no_mda1(dummy_bilevel_scenario) -> None:
    """Test that a warm start chain is built even if the process does not include any
    MDA1.
    """
    assert isinstance(dummy_bilevel_scenario.formulation.chain, MDOWarmStartedChain)


@pytest.mark.parametrize("scenario_formulation_name", ["BiLevel"], indirect=True)
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
    sobieski_bilevel_scenario,
) -> None:
    """Check that the variables from both MDAs are being considered in the warm
    start."""
    scenario = sobieski_bilevel_scenario()
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
        "z",
        design_space.filter(["y"], copy=True),
        formulation_name=sub_scenario_formulation,
        name="FooScenario",
    )
    sub_scenario.set_algorithm(algo_name="NLOPT_COBYLA", max_iter=2)
    scenario = MDOScenario(
        [sub_scenario],
        "z",
        design_space.filter(["x"]),
        formulation_name=scenario_formulation,
        **settings,
    )
    scenario.execute(algo_name="NLOPT_COBYLA", max_iter=2)
    sub_scenarios_log_level = settings.get("sub_scenarios_log_level")
    if sub_scenarios_log_level == logging.WARNING:
        assert "Start FooScenario execution" not in caplog.text
    else:
        assert "Start FooScenario execution" in caplog.text


def test_remove_couplings_from_ds(sobieski_sub_scenarios, caplog) -> None:
    """Check the removal of strong couplings for the design space."""
    formulation = BiLevel(
        [*sobieski_sub_scenarios, SobieskiMission()],
        "y_4",
        SobieskiProblem().design_space,
    )
    for strong_coupling in ["y_12", "y_21", "y_23", "y_31", "y_32"]:
        assert strong_coupling not in formulation.design_space
    assert (
        "The coupling variable y_12 was removed from the design space." in caplog.text
    )


@pytest.mark.parametrize(
    ("scenario", "subscenario"),
    [
        (
            sobieski_bilevel_bcd_scenario,
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
            sobieski_bilevel_scenario,
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
            adapter.scenario.formulation.optimization_problem.design_space.variable_names
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
        "z",
        design_space.filter(["y"], copy=True),
        formulation_name=sub_scenario_formulation,
        name="FooScenario",
    )
    sub_scenario_1.set_algorithm(algo_name="NLOPT_COBYLA", max_iter=2)

    sub_scenario_2 = MDOScenario(
        [AnalyticDiscipline({"c": "(x+b)**2"}, "bar")],
        "c",
        design_space.filter(["b"], copy=True),
        formulation_name=sub_scenario_formulation,
        name="BarScenario",
    )
    sub_scenario_2.set_algorithm(algo_name="NLOPT_COBYLA", max_iter=2)

    scenario = MDOScenario(
        [
            sub_scenario_1,
            sub_scenario_2,
            AnalyticDiscipline({"qux": "baz"}),
            AnalyticDiscipline({"x": "x"}),
        ],
        "z",
        design_space.filter(["x", "baz"]),
        formulation_name=scenario_formulation,
        apply_cstr_tosub_scenarios=False,
    )
    assert "x" not in scenario.formulation.chain._variable_names_to_warm_start
    assert "baz" not in scenario.formulation.chain._variable_names_to_warm_start


def test_bcd_mda(sobieski_bilevel_bcd_scenario):
    """Test that a BCD MDA is created and included in the chain."""
    scenario = sobieski_bilevel_bcd_scenario()
    assert scenario.formulation.bcd_mda
    assert scenario.formulation.bcd_mda == scenario.formulation.chain.disciplines[1]


@pytest.mark.parametrize("keep_opt_history", [True, False])
def test_keep_opt_history(sobieski_bilevel_scenario, keep_opt_history) -> None:
    """Test the keep_opt_history setting."""
    scenario = sobieski_bilevel_scenario(keep_opt_history=keep_opt_history)
    scenario.execute(algo_name="NLOPT_COBYLA", max_iter=2)
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
    tmp_wd, sobieski_bilevel_scenario, save_opt_history, naming
) -> None:
    """Test the save_opt_history and naming settings."""
    scenario = sobieski_bilevel_scenario(
        save_opt_history=save_opt_history, naming=naming
    )
    scenario.execute(algo_name="NLOPT_COBYLA", max_iter=2)
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
