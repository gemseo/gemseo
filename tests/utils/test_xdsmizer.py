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
#        :author: Remi Lafage
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pytest

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.core.chain import MDOChain
from gemseo.core.chain import MDOParallelChain
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.mdo_scenario import MDODiscipline
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from gemseo.utils.xdsmizer import EdgeType
from gemseo.utils.xdsmizer import NodeType
from gemseo.utils.xdsmizer import XDSMizer
from gemseo.utils.xdsmizer import expand

from ..mda.test_mda import analytic_disciplines_from_desc

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.core.scenario import Scenario


def build_sobieski_scenario(
    formulation: str = "MDF", **options: dict[str, Any]
) -> MDOScenario:
    """Scenario based on Sobieski case.

    Args:
        formulation: The name of the formulation.
        options: Any options for the scenario.

    Returns
        The scenario.
    """
    disciplines = [
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
        SobieskiStructure(),
    ]
    sc = MDOScenario(
        disciplines,
        formulation=formulation,
        objective_name="y_4",
        design_space=SobieskiDesignSpace(),
        **options,
    )

    sc.formulation.minimize_objective = False
    return sc


@pytest.fixture()
def options(tmp_path):
    """The options to be passed to the xdsm generation.

    Args:
        file_name: The name of file.
    """

    def _options(file_name):
        return {
            "directory_path": tmp_path,
            "file_name": file_name,
            "save_html": False,
            "save_json": True,
            "save_pdf": True,
            "pdf_build": False,
        }

    return _options


def elementary_discipline(input_name, output_name):
    """Build an elementary analytic discipline from input and output names.

    The discipline is such as ``output_name = input_name``.
    """
    return create_discipline(
        "AnalyticDiscipline", expressions={output_name: input_name}
    )


def test_expand(tmp_path) -> None:
    """Test the workflow expand."""
    mda = ExecutionSequenceFactory.atom(MDODiscipline("mda"))
    d1 = ExecutionSequenceFactory.atom(MDODiscipline("d1"))
    d2 = ExecutionSequenceFactory.atom(MDODiscipline("d2"))
    to_id = {mda: "mda", d1: "d1", d2: "d2"}

    serial_seq = ExecutionSequenceFactory.serial([]).extend(d1)
    loop_seq = ExecutionSequenceFactory.loop(mda, serial_seq)

    assert expand(loop_seq, to_id) == ["mda", ["d1"]]
    assert expand(ExecutionSequenceFactory.serial([]), to_id) == []
    assert expand(serial_seq, to_id) == ["d1"]

    parallel_seq = ExecutionSequenceFactory.parallel([]).extend(d1)
    parallel_seq.extend(d2)

    loop_seq = ExecutionSequenceFactory.loop(mda, parallel_seq)
    assert expand(loop_seq, to_id) == ["mda", [{"parallel": ["d1", "d2"]}]]

    with pytest.raises(TypeError):
        expand("a_bad_exec_seq", to_id)


def test_xdsmize_mdf(options) -> None:
    """Test xdsmization of Sobieski problem solved with MDF with and without
    constraint."""

    scenario = build_sobieski_scenario(inner_mda_name="MDAGaussSeidel")
    assert_xdsm(scenario, **options("xdsmized_sobieski_mdf"))

    # with constraints
    scenario.add_constraint(["g_1", "g_2", "g_3"], "ineq")
    assert_xdsm(scenario, **options("xdsmized_sobieski_cstr_mdf"))

    xdsmizer = XDSMizer(scenario)
    with pytest.raises(ValueError):
        xdsmizer._find_atom(Sellar1())


def test_xdsmize_idf(options) -> None:
    """Test xdsmization of Sobieski problem solved with IDF with and without
    constraint."""
    scenario = build_sobieski_scenario("IDF")
    assert_xdsm(scenario, **options("xdsmized_sobieski_idf"))

    # with constraints
    for c_name in ["g_1", "g_2", "g_3"]:
        scenario.add_constraint(c_name, "ineq")
    assert_xdsm(scenario, **options("xdsmized_sobieski_cstr_idf"))


def test_xdsmize_bilevel(options) -> None:
    """Test xdsmization of Sobieski problem solved with bilevel."""

    design_space = SobieskiDesignSpace()
    propulsion = SobieskiPropulsion()
    aerodynamics = SobieskiAerodynamics()
    structure = SobieskiStructure()
    mission = SobieskiMission()

    algo_options = {
        "xtol_rel": 1e-7,
        "xtol_abs": 1e-7,
        "ftol_rel": 1e-7,
        "ftol_abs": 1e-7,
        "ineq_tolerance": 1e-4,
        "eq_tolerance": 1e-2,
    }
    sub_sc_opts = {"max_iter": 100, "algo": "SLSQP", "algo_options": algo_options}
    sc_prop = MDOScenario(
        disciplines=[propulsion],
        formulation="DisciplinaryOpt",
        objective_name="y_34",
        design_space=deepcopy(design_space).filter("x_3"),
        name="PropulsionScenario",
    )
    sc_prop.default_inputs = sub_sc_opts
    sc_prop.add_constraint("g_3", constraint_type="ineq")

    sc_aero = MDOScenario(
        disciplines=[aerodynamics],
        formulation="DisciplinaryOpt",
        objective_name="y_24",
        design_space=deepcopy(design_space).filter("x_2"),
        name="AerodynamicsScenario",
        maximize_objective=True,
    )
    sc_aero.default_inputs = sub_sc_opts
    sc_aero.add_constraint("g_2", constraint_type="ineq")

    sc_str = MDOScenario(
        disciplines=[structure],
        formulation="DisciplinaryOpt",
        objective_name="y_11",
        design_space=deepcopy(design_space).filter("x_1"),
        name="StructureScenario",
        maximize_objective=True,
    )
    sc_str.add_constraint("g_1", constraint_type="ineq")
    sc_str.default_inputs = sub_sc_opts

    sub_disciplines = [sc_prop, sc_aero, sc_str, mission]

    design_space = deepcopy(design_space).filter("x_shared")
    system_scenario = MDOScenario(
        sub_disciplines,
        formulation="BiLevel",
        objective_name="y_4",
        design_space=design_space,
        maximize_objective=True,
        apply_cstr_tosub_scenarios=False,
        parallel_scenarios=False,
        apply_cstr_to_system=True,
        n_processes=5,
    )
    system_scenario.add_constraint(["g_1", "g_2", "g_3"], "ineq")
    assert_xdsm(system_scenario, **options("xdsmized_sobieski_bilevel"))

    system_scenario_par = MDOScenario(
        sub_disciplines,
        formulation="BiLevel",
        objective_name="y_4",
        design_space=design_space,
        maximize_objective=True,
        apply_cstr_tosub_scenarios=False,
        apply_cstr_to_system=True,
        parallel_scenarios=True,
        n_processes=4,
        use_threading=True,
    )
    system_scenario_par.add_constraint(["g_1", "g_2", "g_3"], "ineq")
    assert_xdsm(system_scenario_par, **options("xdsmized_sobieski_bilevel_parallel"))


def test_xdsmize_nested_chain(options) -> None:
    """Test the XDSM representation of nested ``MDOChain``s.

    Here, we build a 3-levels nested chain.
    """

    def get_name(x: int) -> str:
        return f"x_{x}"

    deep_chain = MDOChain([
        elementary_discipline(get_name(1), get_name(2)),
        elementary_discipline(get_name(2), get_name(3)),
    ])

    inter_chain = MDOChain([
        deep_chain,
        elementary_discipline(get_name(3), get_name(4)),
    ])

    main_chain = [inter_chain, elementary_discipline(get_name(4), get_name(5))]

    design_space = DesignSpace()
    design_space.add_variable(get_name(1))

    nested_chains = MDOScenario(
        main_chain,
        "DisciplinaryOpt",
        get_name(5),
        design_space,
    )

    assert_xdsm(nested_chains, **options("xdsmized_nested_chains"))


@pytest.mark.parametrize("mda_class", [MDAGaussSeidel, MDAJacobi, MDANewtonRaphson])
def test_xdsmize_nested_mda(options, mda_class) -> None:
    """Test the XDSM representation of nested ``MDA``s.

    Here, we build a 2-levels nested mda.
    """

    disciplines = create_disciplines_from_desc([
        (
            "D1",
            ["y2", "y3", "x0"],
            ["y1"],
        ),
        (
            "D2",
            ["y1", "y3", "x0"],
            ["y2", "y2_bis"],
        ),
        (
            "D3",
            ["y2_bis", "x0"],
            ["y3"],
        ),
    ])

    inner_mda = mda_class([disciplines[0], disciplines[1]])

    design_space = DesignSpace()
    design_space.add_variable("x0")

    scenario = MDOScenario(
        [disciplines[2], inner_mda],
        "MDF",
        "y3",
        design_space,
    )

    assert_xdsm(scenario, **options(f"xdsmized_nested_mda_{mda_class.__name__}"))


def test_xdsmize_nested_adapter(options) -> None:
    """Test the XDSM representation of nested ``MDOScenarioAdapter``s.

    Here, we build a 4-levels nested adapter.
    """

    disciplines = create_disciplines_from_desc([
        ("D1", ["x0", "x1"], ["z1", "y1"]),
        ("D2", ["x0", "x2", "y3", "y1"], ["z2", "y2"]),
        ("D3", ["x0", "x3", "y2", "z4"], ["y3", "z3"]),
        ("D4", ["x0", "x4", "x3"], ["z4"]),
    ])

    # -- level 3
    ds_depth3 = DesignSpace()
    ds_depth3.add_variable("x4")

    sce_depth3 = create_scenario(
        [disciplines[3]],
        "MDF",
        "z4",
        ds_depth3,
    )

    # -- level 2
    ds_depth2 = DesignSpace()
    ds_depth2.add_variable("x3")

    adapter_depth2 = MDOScenarioAdapter(
        sce_depth3, input_names=["x3"], output_names=["z4"]
    )

    sce_depth2 = create_scenario(
        [disciplines[2], adapter_depth2],
        "MDF",
        "z3",
        ds_depth2,
    )

    # -- level 1
    ds_depth1 = DesignSpace()
    ds_depth1.add_variable("x2")

    adapter_depth1 = MDOScenarioAdapter(
        sce_depth2, input_names=["y2"], output_names=["y3"]
    )

    sce_depth1 = create_scenario(
        [disciplines[1], adapter_depth1],
        "MDF",
        "z2",
        ds_depth1,
    )

    # -- level 0
    ds_depth0 = DesignSpace()
    ds_depth0.add_variable("x0")

    adapter_depth0 = MDOScenarioAdapter(
        sce_depth1, input_names=["x0", "y1"], output_names=["z2"]
    )

    sce_glob = create_scenario(
        [disciplines[0], adapter_depth0],
        "MDF",
        "z2",
        ds_depth0,
    )

    assert_xdsm(sce_glob, **options("xdsmized_nested_adapter"))


def test_xdsmize_disciplinary_opt_with_adapter(options) -> None:
    """Test that an XDSM with a DisciplinaryOpt formulation involving a single adapter
    is generated correctly."""

    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0)

    disciplines = create_disciplines_from_desc([
        ("D1", ["x", "n"], ["y"]),
        ("D2", ["y"], ["z"]),
    ])

    scenario = create_scenario(
        disciplines,
        "MDF",
        "z",
        design_space,
        scenario_type="MDO",
    )

    adapter = MDOScenarioAdapter(
        scenario,
        input_names=["n"],
        output_names=["x", "y", "z"],
    )

    design_space_discrete = DesignSpace()
    design_space_discrete.add_variable("n", var_type="integer")

    top_scenario = create_scenario(
        [adapter],
        "DisciplinaryOpt",
        "z",
        design_space_discrete,
        scenario_type="DOE",
    )

    assert_xdsm(top_scenario, **options("xdsmized_disciplinary_opt_adapter"))


def test_xdsmize_nested_parallel_chain(options) -> None:
    """Test the XDSM representation of nested ``MDOParallelChain``s.

    Here, we build a 3-levels nested chain.
    """

    def get_name(x: int) -> str:
        return f"x_{x}"

    beg_chain = elementary_discipline(get_name(1), get_name(2))

    deep_chain = MDOParallelChain([
        elementary_discipline(get_name(2), get_name(3)),
        elementary_discipline(get_name(2), get_name(4)),
    ])

    inter_chain = MDOParallelChain([
        deep_chain,
        elementary_discipline(get_name(4), get_name(5)),
    ])

    design_space = DesignSpace()
    design_space.add_variable(get_name(1))

    nested_chains = MDOScenario(
        [beg_chain, inter_chain],
        "DisciplinaryOpt",
        get_name(5),
        design_space,
    )

    assert_xdsm(nested_chains, **options("xdsmized_nested_parallel_chains"))


def test_xdsmize_chain_of_parallel_chain(options) -> None:
    """Test the XDSM representation of nested ``MDOParallelChain``s.

    Here, we build a 3-levels nested chain.
    """

    def get_name(x: int) -> str:
        return f"x_{x}"

    beg_chain = elementary_discipline(get_name(1), get_name(2))

    par_chain = MDOParallelChain([
        elementary_discipline(get_name(2), get_name(3)),
        MDOChain([
            elementary_discipline(get_name(2), get_name(4)),
            elementary_discipline(get_name(4), get_name(5)),
        ]),
    ])

    end_chain = MDOChain([
        elementary_discipline(get_name(3), get_name(6)),
        elementary_discipline(get_name(5), get_name(7)),
    ])

    design_space = DesignSpace()
    design_space.add_variable(get_name(1))

    sce = MDOScenario(
        [beg_chain, par_chain, end_chain],
        "DisciplinaryOpt",
        get_name(7),
        design_space,
    )
    sce.add_constraint(get_name(5))

    assert_xdsm(sce, **options("xdsmized_chain_of_parallel_chain"))


def assert_xdsm(scenario: Scenario, **options: Mapping[str, Any]) -> None:
    """Build and check the XDSM representation generated from a scenario.

    Check both html and tikz generation.

    Args:
        scenario: The scenario from which the XDSM is generated.
        **options: The options for the XDSMizer.
    """
    fname = options["file_name"]
    tmp_path = options["directory_path"]

    xdsmizer = XDSMizer(scenario)
    xdsmizer.run(**options)

    assert_xdsm_json_file_ok(str(tmp_path / fname), fname)
    assert_xdsm_tikz_file_ok(str(tmp_path / fname), fname)


def assert_xdsm_json_file_ok(generated_file: str, ref_file: str) -> None:
    """Check the equality of two XDSM files (json).

    Args:
        generated_file: The name of the generated file.
        ref_file: The name of the reference file.
            This reference file must be located into the ``data`` directory.
    """
    current_dir = Path(__file__).parent
    ref_filepath = (current_dir / "data" / ref_file).with_suffix(".json")

    assert ref_filepath.exists(), (
        f"Reference {ref_filepath!s} not found in data " f"directory."
    )

    xdsm_str = ref_filepath.read_text()
    expected = json.loads(xdsm_str)

    new_filepath = Path(generated_file).with_suffix(".json")
    assert new_filepath.exists(), f"Generated {new_filepath!s} not found."

    xdsm_str = new_filepath.read_text()
    xdsm_json = json.loads(xdsm_str)

    assert_xdsm_equal(expected, xdsm_json)


def assert_xdsm_tikz_file_ok(generated_file: str, ref_file: str) -> None:
    """Check the equality of two XDSM files (tikz) for pdf generation.

    Args:
        generated_file: The name of the generated file.
        ref_file: The name of the reference file.
            This reference file must be located into the ``data`` directory.
    """
    current_dir = Path(__file__).parent
    ref_filepath = (current_dir / "data" / ref_file).with_suffix(".tikz")

    assert ref_filepath.exists(), (
        f"Reference {ref_filepath!s} not found in data " f"directory."
    )

    expected_tikz = remove_tikz_input(ref_filepath.read_text())

    new_filepath = Path(generated_file).with_suffix(".tikz")
    assert new_filepath.exists(), f"Generated {new_filepath!s} not found."

    generated_tikz = remove_tikz_input(new_filepath.read_text())

    assert generated_tikz == expected_tikz


def remove_tikz_input(text: str) -> str:
    """Remove the `input` tag from tikz file.

    Args:
        text: The text contained into the tikz file.
    """
    return re.sub(r"\\input{(.*)}", lambda match: "", text)


def assert_xdsm_equal(expected: dict[str, Any], generated: dict[str, Any]) -> None:
    """Check the equality of two XDSM structures.

    Args:
        expected: The expected XDSM structure to be compared with.
        generated: The generated XDSM structure.
    """
    assert sorted(expected.keys()) == sorted(generated.keys())

    for key in expected:
        assert_level_xdsm_equal(expected[key], generated[key])


def assert_level_xdsm_equal(
    expected: Mapping[str, NodeType | EdgeType],
    generated: Mapping[str, NodeType | EdgeType],
) -> None:
    """Check the equality of ``nodes`` and ``edges`` in two different XDSM structures.

    Args:
        expected: The expected data to be compared with.
        generated: The generated data.
    """
    assert len(expected["nodes"]) == len(generated["nodes"])

    for expected_node in expected["nodes"]:
        found = False
        for node in generated["nodes"]:
            if node["id"] == expected_node["id"]:
                assert expected_node["name"] == node["name"]
                assert expected_node["type"] == node["type"]
                found = True
        assert found, f"Node {expected_node!s} not found."

    for expected_edge in expected["edges"]:
        found = False
        for edge in generated["edges"]:
            if (
                edge["from"] == expected_edge["from"]
                and edge["to"] == expected_edge["to"]
            ):
                assert set(expected_edge["name"].split(", ")) == set(
                    edge["name"].split(", ")
                )
                found = True
        assert found, f"Edge {expected_edge!s} not found."
    assert expected["workflow"] == generated["workflow"]


def test_xdsmize_mdf_mdoparallelchain(options) -> None:
    """Test the XDSM representation of an MDF including an MDOParallelChain.

    In this case, the two MDAGaussSeidel created in the MDAChain must be parallel
    """
    disciplines = analytic_disciplines_from_desc((
        {"a": "x"},
        {"y1": "x1", "b": "a+1"},
        {"x1": "1.-0.3*y1"},
        {"y2": "x2", "c": "a+2"},
        {"x2": "1.-0.3*y2"},
    ))
    design_space = DesignSpace()
    design_space.add_variable("x")
    mdachain_parallel_options = {"use_threading": True, "n_processes": 2}
    scenario = MDOScenario(
        disciplines,
        "MDF",
        "y2",
        design_space,
        mdachain_parallelize_tasks=True,
        mdachain_parallel_options=mdachain_parallel_options,
        inner_mda_name="MDAGaussSeidel",
    )

    assert_xdsm(scenario, **options("xdsmized_mdf_mdoparallelchain"))


@pytest.mark.parametrize("directory_path", [".", Path("bar")])
@pytest.mark.parametrize("file_name", [None, "foo"])
@pytest.mark.parametrize("save_html", [False, True])
def test_run_return(tmp_wd, directory_path, file_name, save_html) -> None:
    """Check the object returned by XDSMizer.run()."""
    directory_path = tmp_wd / directory_path

    design_space = DesignSpace()
    design_space.add_variable("x")
    discipline = AnalyticDiscipline({"y": "x"})
    scenario = MDOScenario([discipline], "DisciplinaryOpt", "y", design_space)

    file_name = file_name or "xdsm"
    xdsm = XDSMizer(scenario).run(
        directory_path=directory_path,
        file_name=file_name,
        save_json=True,
        save_html=save_html,
    )
    json_path = directory_path / f"{file_name}.json"

    with json_path.open("r") as f:
        assert xdsm.json_schema == f.read()

    html_file_name = f"{file_name}.html"
    if not save_html:
        assert xdsm.html_file_path is None
    elif directory_path != ".":
        # The output directory containing the HTML is given by the user.
        assert xdsm.html_file_path == directory_path / html_file_name
    else:
        # The output directory is temporary.
        html_file_path = xdsm.html_file_path
        assert html_file_path.exists()
        assert html_file_path.name == html_file_name
