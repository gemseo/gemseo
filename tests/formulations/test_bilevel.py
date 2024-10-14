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

import pytest

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.core.chains.warm_started_chain import MDOWarmStartedChain
from gemseo.core.discipline import Discipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.utils import get_sub_disciplines
from gemseo.formulations.bilevel import BiLevel
from gemseo.problems.mdo.aerostructure.aerostructure_design_space import (
    AerostructureDesignSpace,
)
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.testing.bilevel_test_helper import create_sobieski_bilevel_scenario
from gemseo.utils.testing.bilevel_test_helper import create_sobieski_sub_scenarios
from tests.core.test_dependency_graph import create_disciplines_from_desc


@pytest.fixture
def sobieski_bilevel_scenario():
    """Fixture from an existing function."""
    return create_sobieski_bilevel_scenario()


@pytest.fixture
def dummy_bilevel_scenario() -> MDOScenario:
    """Create a dummy BiLevel scenario.

    It has to be noted that there is no strongly coupled discipline in this example.
    It implies that MDA1 will not be created. Yet, MDA2 will be created,
    as it is built with all the sub-disciplines passed to the BiLevel formulation.

    Returns: A dummy BiLevel MDOScenario.
    """
    disc_expressions = {
        "disc_1": (["x_1"], ["a"]),
        "disc_2": (["a", "x_2"], ["b"]),
        "disc_3": (["x", "x_3", "b"], ["obj"]),
    }
    discipline_1, discipline_2, discipline_3 = create_disciplines_from_desc(
        disc_expressions
    )

    system_design_space = create_design_space()
    system_design_space.add_variable("x_3")

    sub_design_space_1 = create_design_space()
    sub_design_space_1.add_variable("x_1")
    sub_scenario_1 = create_scenario(
        [discipline_1, discipline_3],
        "MDF",
        "obj",
        sub_design_space_1,
    )

    sub_design_space_2 = create_design_space()
    sub_design_space_2.add_variable("x_2")
    sub_scenario_2 = create_scenario(
        [discipline_2, discipline_3],
        "MDF",
        "obj",
        sub_design_space_2,
    )

    return create_scenario(
        [sub_scenario_1, sub_scenario_2],
        "BiLevel",
        "obj",
        system_design_space,
    )


def test_execute(sobieski_bilevel_scenario) -> None:
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


def test_bilevel_aerostructure() -> None:
    """Test the Bi-level formulation on the aero-structure problem."""
    algo_options = {
        "xtol_rel": 1e-8,
        "xtol_abs": 1e-8,
        "ftol_rel": 1e-8,
        "ftol_abs": 1e-8,
        "ineq_tolerance": 1e-5,
        "eq_tolerance": 1e-3,
    }

    aero_formulas = {
        "drag": "0.1*((sweep/360)**2 + 200 + thick_airfoils**2 - thick_airfoils - "
        "4*displ)",
        "forces": "10*sweep + 0.2*thick_airfoils - 0.2*displ",
        "lift": "(sweep + 0.2*thick_airfoils - 2.*displ)/3000.",
    }
    aerodynamics = create_discipline(
        "AnalyticDiscipline", name="Aerodynamics", expressions=aero_formulas
    )
    struc_formulas = {
        "mass": "4000*(sweep/360)**3 + 200000 + 100*thick_panels + 200.0*forces",
        "reserve_fact": "-3*sweep - 6*thick_panels + 0.1*forces + 55",
        "displ": "2*sweep + 3*thick_panels - 2.*forces",
    }
    structure = create_discipline(
        "AnalyticDiscipline", name="Structure", expressions=struc_formulas
    )
    mission_formulas = {"range": "8e11*lift/(mass*drag)"}
    mission = create_discipline(
        "AnalyticDiscipline", name="Mission", expressions=mission_formulas
    )
    sub_scenario_options = {
        "max_iter": 2,
        "algo": "NLOPT_SLSQP",
        "algo_options": algo_options,
    }
    design_space_ref = AerostructureDesignSpace()

    design_space_aero = design_space_ref.filter(["thick_airfoils"], copy=True)
    aero_scenario = create_scenario(
        [aerodynamics, mission],
        "DisciplinaryOpt",
        "range",
        design_space_aero,
        maximize_objective=True,
    )
    aero_scenario.default_input_data = sub_scenario_options

    design_space_struct = design_space_ref.filter(["thick_panels"], copy=True)
    struct_scenario = create_scenario(
        [structure, mission],
        "DisciplinaryOpt",
        "range",
        design_space_struct,
        maximize_objective=True,
    )
    struct_scenario.default_input_data = sub_scenario_options

    design_space_system = design_space_ref.filter(["sweep"], copy=True)
    system_scenario = create_scenario(
        [aero_scenario, struct_scenario, mission],
        "BiLevel",
        "range",
        design_space_system,
        maximize_objective=True,
        main_mda_name="MDAJacobi",
        tolerance=1e-8,
    )
    system_scenario.add_constraint("reserve_fact", constraint_type="ineq", value=0.5)
    system_scenario.add_constraint("lift", value=0.5)
    system_scenario.execute(algo="NLOPT_COBYLA", max_iter=5, algo_options=algo_options)


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
    assert "b" in disciplines[0].io.input_grammar.names
    assert "a" in disciplines[0].io.output_grammar.names

    # a is a weak coupling of all the disciplines,
    # and it is in the top-level inputs of the second adapter
    assert "a" in disciplines[1].io.input_grammar.names

    # a is a weak coupling of all the disciplines,
    # and is in the top-level inputs of the second adapter
    assert "b" in disciplines[1].io.output_grammar.names


def test_bilevel_mda_getter(dummy_bilevel_scenario) -> None:
    """Test that the user can access the MDA1 and MDA2."""
    # In the Dummy scenario, there's not strongly coupled disciplines -> No MDA1
    assert dummy_bilevel_scenario.formulation.mda1 is None
    assert "obj" in dummy_bilevel_scenario.formulation.mda2.io.output_grammar.names


def test_bilevel_mda_setter(dummy_bilevel_scenario) -> None:
    """Test that the user cannot modify the MDA1 and MDA2 after instantiation."""
    discipline = create_discipline("SellarSystem")
    with pytest.raises(AttributeError):
        dummy_bilevel_scenario.formulation.mda1 = discipline
    with pytest.raises(AttributeError):
        dummy_bilevel_scenario.formulation.mda2 = discipline


def test_get_sub_disciplines(sobieski_bilevel_scenario) -> None:
    """Test the get_sub_disciplines method with the BiLevel formulation.

    Args:
        sobieski_bilevel_scenario: Fixture to instantiate a Sobieski BiLevel Scenario.
    """
    scenario = sobieski_bilevel_scenario()
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


def test_bilevel_warm_start(sobieski_bilevel_scenario) -> None:
    """Test the warm start of the BiLevel chain.

    Args:
        sobieski_bilevel_scenario: Fixture to instantiate a Sobieski BiLevel Scenario.
    """
    scenario = sobieski_bilevel_scenario()
    scenario.formulation.chain.set_cache(Discipline.CacheType.MEMORY_FULL)
    bilevel_chain_cache = scenario.formulation.chain.cache
    scenario.formulation.chain.disciplines[0].set_cache(
        Discipline.CacheType.MEMORY_FULL
    )
    mda1_cache = scenario.formulation.chain.disciplines[0].cache
    scenario.execute(algo="NLOPT_COBYLA", max_iter=3)
    mda1_inputs = [entry.inputs for entry in mda1_cache.get_all_entries()]
    chain_outputs = [entry.outputs for entry in bilevel_chain_cache.get_all_entries()]

    assert mda1_inputs[1]["y_21"] == chain_outputs[0]["y_21"]
    assert (mda1_inputs[1]["y_12"] == chain_outputs[0]["y_12"]).all()
    assert mda1_inputs[2]["y_21"] == chain_outputs[1]["y_21"]
    assert (mda1_inputs[2]["y_12"] == chain_outputs[1]["y_12"]).all()


def test_bilevel_warm_start_no_mda1(dummy_bilevel_scenario) -> None:
    """Test that a warm start chain is built even if the process does not include any
    MDA1.

    Args:
        dummy_bilevel_scenario: Fixture to instantiate a dummy weakly
            coupled scenario.
    """
    assert isinstance(dummy_bilevel_scenario.formulation.chain, MDOWarmStartedChain)


def test_bilevel_get_variable_names_to_warm_start_without_mdas(
    dummy_bilevel_scenario, monkeypatch
) -> None:
    """ " Check that the warm start properly considers the adapter variables when there
    are no MDAs."""

    def _no_mda2(*args, **kwargs):
        return None

    scenario = dummy_bilevel_scenario
    monkeypatch.setattr(scenario.formulation, "_mda2", _no_mda2())
    variables = []
    for adapter in scenario.formulation.scenario_adapters:
        variables.extend(adapter.io.output_grammar.names)
    assert sorted(set(variables)) == sorted(
        scenario.formulation._get_variable_names_to_warm_start()
    )


def test_test_bilevel_get_variable_names_to_warm_start_from_mdas(
    sobieski_bilevel_scenario,
) -> None:
    """ " Check that the variables from both MDAs are being considered in the warm
    start."""
    scenario = sobieski_bilevel_scenario()
    for variable in scenario.formulation._mda1.io.output_grammar.names:
        assert variable in scenario.formulation._get_variable_names_to_warm_start()
    for variable in scenario.formulation._mda2.io.output_grammar.names:
        assert variable in scenario.formulation._get_variable_names_to_warm_start()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"sub_scenarios_log_level": None},
        {"sub_scenarios_log_level": logging.INFO},
        {"sub_scenarios_log_level": logging.WARNING},
    ],
)
def test_scenario_log_level(caplog, options) -> None:
    """Check scenario_log_level."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_variable("y", lower_bound=0.0, upper_bound=1.0, value=0.5)
    sub_scenario = MDOScenario(
        [AnalyticDiscipline({"z": "(x+y)**2"})],
        "DisciplinaryOpt",
        "z",
        design_space.filter(["y"], copy=True),
        name="FooScenario",
    )
    sub_scenario.default_input_data = {"algo": "NLOPT_COBYLA", "max_iter": 2}
    scenario = MDOScenario(
        [sub_scenario], "BiLevel", "z", design_space.filter(["x"]), **options
    )
    scenario.execute(algo="NLOPT_COBYLA", max_iter=2)
    sub_scenarios_log_level = options.get("sub_scenarios_log_level")
    if sub_scenarios_log_level == logging.WARNING:
        assert "Start FooScenario execution" not in caplog.text
    else:
        assert "Start FooScenario execution" in caplog.text


@pytest.fixture
def sobieski_sub_scenarios() -> tuple[MDOScenario, MDOScenario, MDOScenario]:
    """Fixture from an existing function."""
    return create_sobieski_sub_scenarios()


def test_remove_couplings_from_ds(sobieski_sub_scenarios) -> None:
    """Check the removal of strong couplings for the design space."""
    formulation = BiLevel(
        [*sobieski_sub_scenarios, SobieskiMission()],
        "y_4",
        SobieskiProblem().design_space,
    )
    for strong_coupling in ["y_12", "y_21", "y_23", "y_31", "y_32"]:
        assert strong_coupling not in formulation.design_space
