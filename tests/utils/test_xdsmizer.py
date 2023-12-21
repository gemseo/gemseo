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
import unittest
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


@pytest.mark.usefixtures("tmp_wd")
class TestXDSMizer(unittest.TestCase):
    """Test XDSM diagram json generation."""

    def test_expand(self):
        """"""
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

        self.assertRaises(Exception, expand, "a_bad_exec_seq", to_id)

    def build_mdo_scenario(self, formulation="MDF", **options):
        """

        :param formulation: Default value = 'MDF')

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

    def test_xdsmize_mdf(self):
        """Test xdsmization of Sobieski problem solved with MDF with and without
        constraint."""
        scenario = self.build_mdo_scenario(inner_mda_name="MDAGaussSeidel")
        options = {
            "directory_path": ".",
            "save_html": False,
            "save_json": True,
            "file_name": "xdsmized_sobieski_mdf",
        }
        assert_xdsm(scenario, **options)

        # with constraints
        options["file_name"] = "xdsmized_sobieski_cstr_mdf"
        scenario.add_constraint(["g_1", "g_2", "g_3"], "ineq")
        assert_xdsm(scenario, **options)

        # without directory_path
        xdsmizer = XDSMizer(scenario)
        xdsmizer.run(save_json=True, file_name="xdsmized_sobieski_mdf")

        self.assertRaises(ValueError, xdsmizer._find_atom, Sellar1())

        xdsmizer.run(file_name="xdsmized_sobieski_mdf")

    def test_xdsmize_idf(self):
        """Test xdsmization of Sobieski problem solved with IDF with and without
        constraint."""
        formulation = "IDF"
        scenario = self.build_mdo_scenario(formulation)
        scenario.xdsmize(save_html=False, save_json=True)
        options = {
            "directory_path": ".",
            "save_html": False,
            "save_json": True,
            "file_name": "xdsmized_sobieski_idf",
        }
        assert_xdsm(scenario, **options)

        # with constraints
        options["file_name"] = "xdsmized_sobieski_cstr_idf"
        for c_name in ["g_1", "g_2", "g_3"]:
            scenario.add_constraint(c_name, "ineq")
        assert_xdsm(scenario, **options)

    def test_xdsmize_bilevel(self):
        """Test xdsmization of Sobieski problem solved with bilevel."""
        design_space = SobieskiDesignSpace()
        # Disciplinary optimization
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
        # Minimize SFC
        sc_prop = MDOScenario(
            disciplines=[propulsion],
            formulation="DisciplinaryOpt",
            objective_name="y_34",
            design_space=deepcopy(design_space).filter("x_3"),
            name="PropulsionScenario",
        )
        sc_prop.default_inputs = sub_sc_opts
        sc_prop.add_constraint("g_3", constraint_type="ineq")

        # Maximize L/D
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

        # Maximize log(aircraft total weight / (aircraft total weight - fuel
        # weight))
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

        # Maximize range (Breguet)
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
        system_scenario.xdsmize(save_json=True)
        options = {
            "directory_path": ".",
            "save_pdf": False,
            "save_html": False,
            "save_json": True,
            "file_name": "xdsmized_sobieski_bilevel",
        }
        assert_xdsm(system_scenario, **options)

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


@pytest.fixture()
def elementary_discipline():
    """Build an elementary analytic discipline from input and output names.

    The discipline is such as ``output_name = input_name``.
    """

    def _elementary_discipline(input_name, output_name):
        return create_discipline(
            "AnalyticDiscipline", expressions={output_name: input_name}
        )

    return _elementary_discipline


def test_xdsmize_nested_chain(tmp_wd, elementary_discipline):
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
        formulation="DisciplinaryOpt",
        objective_name=get_name(5),
        design_space=design_space,
    )

    options = {
        "save_html": False,
        "save_json": True,
        "file_name": "xdsmized_nested_chains",
    }

    assert_xdsm(nested_chains, **options)


def test_xdsmize_nested_mda(tmp_wd):
    """Test the XDSM representation of nested ``MDA``s.

    Here, we build a 2-levels nested mda with Jacobi and GaussSeidel.
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

    inner_mda = MDAGaussSeidel([disciplines[0], disciplines[1]])

    design_space = DesignSpace()
    design_space.add_variable("x0")

    scenario = MDOScenario(
        [disciplines[2], inner_mda],
        formulation="MDF",
        objective_name="y3",
        design_space=design_space,
    )

    options = {
        "save_html": False,
        "save_json": True,
        "file_name": "xdsmized_nested_mda",
    }

    assert_xdsm(scenario, **options)


def test_xdsmize_nested_adapter(tmp_wd):
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
        formulation="MDF",
        objective_name="z4",
        design_space=ds_depth3,
    )

    # -- level 2
    ds_depth2 = DesignSpace()
    ds_depth2.add_variable("x3")

    adapter_depth2 = MDOScenarioAdapter(
        sce_depth3, input_names=["x3"], output_names=["z4"]
    )

    sce_depth2 = create_scenario(
        [disciplines[2], adapter_depth2],
        formulation="MDF",
        objective_name="z3",
        design_space=ds_depth2,
    )

    # -- level 1
    ds_depth1 = DesignSpace()
    ds_depth1.add_variable("x2")

    adapter_depth1 = MDOScenarioAdapter(
        sce_depth2, input_names=["y2"], output_names=["y3"]
    )

    sce_depth1 = create_scenario(
        [disciplines[1], adapter_depth1],
        formulation="MDF",
        objective_name="z2",
        design_space=ds_depth1,
    )

    # -- level 0
    ds_depth0 = DesignSpace()
    ds_depth0.add_variable("x0")

    adapter_depth0 = MDOScenarioAdapter(
        sce_depth1, input_names=["x0", "y1"], output_names=["z2"]
    )

    sce_glob = create_scenario(
        [disciplines[0], adapter_depth0],
        formulation="MDF",
        objective_name="z2",
        design_space=ds_depth0,
    )

    options = {
        "save_html": False,
        "save_json": True,
        "file_name": "xdsmized_nested_adapter",
    }
    assert_xdsm(sce_glob, **options)


def test_xdsmize_disciplinary_opt_with_adapter(tmp_wd):
    """Test that an XDSM with a DisciplinaryOpt formulation involving a single adapter
    is generated correctly."""

    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0)

    disciplines = create_disciplines_from_desc([
        ("D1", ["x", "n"], ["y"]),
        ("D2", ["y"], ["z"]),
    ])

    scenario = create_scenario(
        disciplines=disciplines,
        formulation="MDF",
        objective_name="z",
        design_space=design_space,
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
        disciplines=[adapter],
        formulation="DisciplinaryOpt",
        objective_name="z",
        design_space=design_space_discrete,
        scenario_type="DOE",
    )

    options = {
        "save_html": False,
        "save_json": True,
        "file_name": "xdsmized_disciplinary_opt_adapter",
    }
    assert_xdsm(top_scenario, **options)


def test_xdsmize_nested_parallel_chain(tmp_wd, elementary_discipline):
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
        formulation="DisciplinaryOpt",
        objective_name=get_name(5),
        design_space=design_space,
    )

    options = {
        "save_html": False,
        "save_json": True,
        "file_name": "xdsmized_nested_parallel_chains",
    }

    assert_xdsm(nested_chains, **options)


def assert_xdsm(scenario: Scenario, **options: Mapping[str, Any]) -> None:
    """Build and check the XDSM representation generated from a scenario.

    Args:
        scenario: The scenario from which the XDSM is generated.
        **options: The options for the XDSMizer.
    """
    fname = options["file_name"]
    xdsmizer = XDSMizer(scenario)
    xdsmizer.run(**options)
    assert_xdsm_file_ok(fname, fname)


def assert_xdsm_file_ok(generated_file: str, ref_file: str) -> None:
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


def test_xdsmize_mdf_mdoparallelchain(tmp_wd):
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
        formulation="MDF",
        objective_name="y2",
        design_space=design_space,
        mdachain_parallelize_tasks=True,
        mdachain_parallel_options=mdachain_parallel_options,
        inner_mda_name="MDAGaussSeidel",
    )

    options = {
        "save_html": False,
        "save_json": True,
        "file_name": "xdsmized_mdf_mdoparallelchain",
    }

    assert_xdsm(scenario, **options)


@pytest.mark.parametrize("directory_path", [".", Path("bar")])
@pytest.mark.parametrize("file_name", [None, "foo"])
@pytest.mark.parametrize("save_html", [False, True])
def test_run_return(tmp_wd, directory_path, file_name, save_html):
    """Check the object returned by XDSMizer.run()."""
    if directory_path != ".":
        (tmp_wd / directory_path).mkdir()

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
    if directory_path == ".":
        json_path = Path(f"{file_name}.json")
    else:
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
