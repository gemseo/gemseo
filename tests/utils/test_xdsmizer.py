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
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.core._process_flow.execution_sequences.execution_sequence import (
    ExecutionSequence,
)
from gemseo.core._process_flow.execution_sequences.loop import LoopExecSequence
from gemseo.core._process_flow.execution_sequences.parallel import ParallelExecSequence
from gemseo.core._process_flow.execution_sequences.sequential import (
    SequentialExecSequence,
)
from gemseo.core.chains.chain import MDOChain
from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.mda_chain import MDAChain
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.problems.mdo.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.discipline import DummyDiscipline
from gemseo.utils.testing.helpers import concretize_classes
from gemseo.utils.xdsm_to_pdf import XDSM
from gemseo.utils.xdsmizer import EdgeType
from gemseo.utils.xdsmizer import NodeType
from gemseo.utils.xdsmizer import XDSMizer
from gemseo.utils.xdsmizer import expand

from ..mda.test_mda import analytic_disciplines_from_desc

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.core.discipline import Discipline
    from gemseo.typing import StrKeyMapping


def build_sobieski_scenario(
    formulation_name: str = "MDF", **formulation_settings: dict[str, Any]
) -> MDOScenario:
    """Scenario based on Sobieski case.

    Args:
        formulation_name: The name of the formulation.
        formulation_settings: The settings for the scenario.

    Returns
        The scenario.
    """
    disciplines = [
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
        SobieskiStructure(),
    ]
    return MDOScenario(
        disciplines,
        "y_4",
        SobieskiDesignSpace(),
        formulation_name=formulation_name,
        maximize_objective=False,
        **formulation_settings,
    )


@pytest.fixture
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
    """Test the process_flow expand."""
    mda = ExecutionSequence(DummyDiscipline("mda"))
    d1 = ExecutionSequence(DummyDiscipline("d1"))
    d2 = ExecutionSequence(DummyDiscipline("d2"))
    to_id = {mda: "mda", d1: "d1", d2: "d2"}

    serial_seq = SequentialExecSequence([])
    serial_seq.extend(d1)
    loop_seq = LoopExecSequence(mda, serial_seq)

    assert expand(loop_seq, to_id) == ["mda", ["d1"]]
    assert expand(SequentialExecSequence([]), to_id) == []
    assert expand(serial_seq, to_id) == ["d1"]

    parallel_seq = ParallelExecSequence([])
    parallel_seq.extend(d1)
    parallel_seq.extend(d2)

    loop_seq = LoopExecSequence(mda, parallel_seq)
    assert expand(loop_seq, to_id) == ["mda", [{"parallel": ["d1", "d2"]}]]

    with pytest.raises(TypeError):
        expand("a_bad_exec_seq", to_id)


def test_xdsmize_mdf(options) -> None:
    """Test xdsmization of Sobieski problem solved with MDF with and without
    constraint."""

    scenario = build_sobieski_scenario(
        main_mda_settings={"inner_mda_name": "MDAGaussSeidel"}
    )
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

    settings_model = SLSQP_Settings(
        max_iter=100,
        xtol_rel=1e-7,
        xtol_abs=1e-7,
        ftol_rel=1e-7,
        ftol_abs=1e-7,
        ineq_tolerance=1e-4,
        eq_tolerance=1e-2,
    )
    sc_prop = MDOScenario(
        [propulsion],
        "y_34",
        deepcopy(design_space).filter("x_3"),
        formulation_name="DisciplinaryOpt",
        name="PropulsionScenario",
    )
    sc_prop.set_algorithm(algo_name="SLSQP", algo_settings_model=settings_model)
    sc_prop.add_constraint("g_3", constraint_type="ineq")

    sc_aero = MDOScenario(
        [aerodynamics],
        "y_24",
        deepcopy(design_space).filter("x_2"),
        formulation_name="DisciplinaryOpt",
        name="AerodynamicsScenario",
        maximize_objective=True,
    )
    sc_prop.set_algorithm(algo_name="SLSQP", algo_settings_model=settings_model)
    sc_aero.add_constraint("g_2", constraint_type="ineq")

    sc_str = MDOScenario(
        [structure],
        "y_11",
        deepcopy(design_space).filter("x_1"),
        formulation_name="DisciplinaryOpt",
        name="StructureScenario",
        maximize_objective=True,
    )
    sc_str.add_constraint("g_1", constraint_type="ineq")
    sc_prop.set_algorithm(algo_name="SLSQP", algo_settings_model=settings_model)

    sub_disciplines = [sc_prop, sc_aero, sc_str, mission]

    design_space = deepcopy(design_space).filter("x_shared")
    system_scenario = MDOScenario(
        sub_disciplines,
        "y_4",
        design_space,
        formulation_name="BiLevel",
        maximize_objective=True,
        apply_cstr_tosub_scenarios=False,
        parallel_scenarios=False,
        apply_cstr_to_system=True,
        main_mda_settings={"n_processes": 5},
    )
    system_scenario.add_constraint(["g_1", "g_2", "g_3"], "ineq")
    assert_xdsm(system_scenario, **options("xdsmized_sobieski_bilevel"))

    system_scenario_par = MDOScenario(
        sub_disciplines,
        "y_4",
        design_space,
        formulation_name="BiLevel",
        maximize_objective=True,
        apply_cstr_tosub_scenarios=False,
        apply_cstr_to_system=True,
        parallel_scenarios=True,
        main_mda_settings={"n_processes": 5, "use_threading": True},
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
        get_name(5),
        design_space,
        formulation_name="DisciplinaryOpt",
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
        "y3",
        design_space,
        formulation_name="MDF",
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
        "z4",
        ds_depth3,
        formulation_name="MDF",
    )

    # -- level 2
    ds_depth2 = DesignSpace()
    ds_depth2.add_variable("x3")

    adapter_depth2 = MDOScenarioAdapter(
        sce_depth3, input_names=["x3"], output_names=["z4"]
    )

    sce_depth2 = create_scenario(
        [disciplines[2], adapter_depth2],
        "z3",
        ds_depth2,
        formulation_name="MDF",
    )

    # -- level 1
    ds_depth1 = DesignSpace()
    ds_depth1.add_variable("x2")

    adapter_depth1 = MDOScenarioAdapter(
        sce_depth2, input_names=["y2"], output_names=["y3"]
    )

    sce_depth1 = create_scenario(
        [disciplines[1], adapter_depth1],
        "z2",
        ds_depth1,
        formulation_name="MDF",
    )

    # -- level 0
    ds_depth0 = DesignSpace()
    ds_depth0.add_variable("x0")

    adapter_depth0 = MDOScenarioAdapter(
        sce_depth1, input_names=["x0", "y1"], output_names=["z2"]
    )

    sce_glob = create_scenario(
        [disciplines[0], adapter_depth0],
        "z2",
        ds_depth0,
        formulation_name="MDF",
    )

    assert_xdsm(sce_glob, **options("xdsmized_nested_adapter"))


def test_xdsmize_disciplinary_opt_with_adapter(options) -> None:
    """Test that an XDSM with a DisciplinaryOpt formulation involving a single adapter
    is generated correctly."""

    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0)

    disciplines = create_disciplines_from_desc([
        ("D1", ["x", "n"], ["y"]),
        ("D2", ["y"], ["z"]),
    ])

    scenario = create_scenario(
        disciplines,
        "z",
        design_space,
        formulation_name="MDF",
        scenario_type="MDO",
    )

    adapter = MDOScenarioAdapter(
        scenario,
        input_names=["n"],
        output_names=["x", "y", "z"],
    )

    design_space_discrete = DesignSpace()
    design_space_discrete.add_variable("n", type_="integer")

    top_scenario = create_scenario(
        [adapter],
        "z",
        design_space_discrete,
        formulation_name="DisciplinaryOpt",
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
        get_name(5),
        design_space,
        formulation_name="DisciplinaryOpt",
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
        get_name(7),
        design_space,
        formulation_name="DisciplinaryOpt",
    )
    sce.add_constraint(get_name(5))

    assert_xdsm(sce, **options("xdsmized_chain_of_parallel_chain"))


def test_xdsmized_parallel_chain_of_mda(options) -> None:
    """Test the XDSM representation of a parallel chain including an MDA."""

    def get_name(x: int) -> str:
        return f"x_{x}"

    par_chain = MDOParallelChain([
        MDOChain([
            elementary_discipline(get_name(1), get_name(2)),
            elementary_discipline(get_name(2), get_name(3)),
        ]),
        MDOChain([
            elementary_discipline(get_name(1), get_name(3)),
            elementary_discipline(get_name(3), get_name(4)),
            MDAGaussSeidel([
                elementary_discipline(get_name(5), get_name(6)),
                elementary_discipline(get_name(4), get_name(5)),
            ]),
        ]),
    ])

    design_space = DesignSpace()
    design_space.add_variable(get_name(1))

    sce = MDOScenario(
        [par_chain, elementary_discipline(get_name(6), get_name(7))],
        get_name(7),
        design_space,
        formulation_name="DisciplinaryOpt",
    )

    assert_xdsm(sce, **options("xdsmized_parallel_chain_of_mda"))


def assert_xdsm(discipline: Discipline, **options: StrKeyMapping) -> None:
    """Build and check the XDSM representation generated from a scenario.

    Check both html and tikz generation.

    Args:
        discipline: The discipline from which the XDSM is generated.
        **options: The options for the XDSMizer.
    """
    fname = options["file_name"]
    tmp_path = options["directory_path"]
    options["pdf_cleanup"] = False

    xdsmizer = XDSMizer(discipline)
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
        f"Reference {ref_filepath!s} not found in data directory."
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
        f"Reference {ref_filepath!s} not found in data directory."
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
    mdachain_parallel_settings = {"use_threading": True, "n_processes": 2}
    scenario = MDOScenario(
        disciplines,
        "y2",
        design_space,
        formulation_name="MDF",
        main_mda_settings={
            "mdachain_parallelize_tasks": True,
            "mdachain_parallel_settings": mdachain_parallel_settings,
            "inner_mda_name": "MDAGaussSeidel",
        },
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
    scenario = MDOScenario(
        [discipline], "y", design_space, formulation_name="DisciplinaryOpt"
    )

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
        assert not xdsm.html_file_path
    elif directory_path != ".":
        # The output directory containing the HTML is given by the user.
        assert xdsm.html_file_path == directory_path / html_file_name
    else:
        # The output directory is temporary.
        html_file_path = xdsm.html_file_path
        assert html_file_path.exists()
        assert html_file_path.name == html_file_name


def test_mda_chain(options) -> None:
    """Test the XDSM representation of an MDAChain."""
    mda_chain = MDAChain([Sellar1(), Sellar2(), SellarSystem()])
    assert_xdsm(mda_chain, **options("xdsmized_mda_chain"))


def test_discipline(options) -> None:
    """Test the XDSM representation of a simple discipline."""
    assert_xdsm(Sellar1(), **options("xdsmized_sellar_1"))


@pytest.mark.parametrize(
    ("cls", "expected"),
    [(MDOScenario, "Optimizer"), (DOEScenario, "DOE")],
)
def test_initial_node_title(cls, expected):
    """Check the title of the initial node."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    discipline = AnalyticDiscipline({"y": "x"})
    with concretize_classes(cls):
        scenario = cls(
            [discipline],
            "y",
            design_space,
            name="foo",
            formulation_name="DisciplinaryOpt",
        )

    xdsmizer = XDSMizer(scenario)
    assert xdsmizer._scenario_node_title == expected


def write(self, file_name, build=True, cleanup=True, quiet=False, outdir="."):
    """Mocks the method pyxdsm.XDSM.write so as not to depend on pdflatex."""
    extensions = {"tex", "tikz"}
    if build:
        extensions.update({"pdf"})
        if not cleanup:
            extensions.update({"aux"})

    for extension in extensions:
        with (Path(outdir) / f"{file_name}.{extension}").open("w") as f:
            f.write("foo")


@pytest.mark.parametrize("pdf_cleanup", [False, True])
@pytest.mark.parametrize("pdf_build", [False, True])
def test_cleanup(tmp_wd, pdf_cleanup, pdf_build, monkeypatch):
    """Check the pdf_cleanup and pdf_build options."""
    xdsmizer = XDSMizer(Sellar1())
    monkeypatch.setattr(XDSM, "write", write)
    xdsmizer.run(
        save_pdf=True,
        pdf_cleanup=pdf_cleanup,
        pdf_build=pdf_build,
        directory_path=tmp_wd,
    )

    assert Path("xdsm.pdf").exists() is pdf_build
    assert Path("xdsm.tikz").exists() is (not pdf_cleanup or pdf_build)
    assert Path("xdsm.tex").exists() is (not pdf_cleanup or pdf_build)
    assert Path("xdsm.aux").exists() is (pdf_build and not pdf_cleanup)
