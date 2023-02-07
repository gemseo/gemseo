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
from typing import Any
from typing import Mapping

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.api import create_discipline
from gemseo.core.chain import MDOChain
from gemseo.core.chain import MDOParallelChain
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.mdo_scenario import MDODiscipline
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from gemseo.utils.xdsmizer import EdgeType
from gemseo.utils.xdsmizer import expand
from gemseo.utils.xdsmizer import NodeType
from gemseo.utils.xdsmizer import XDSMizer
from gemseo.utils.xdsmizer import XdsmType

from ..mda.test_mda import analytic_disciplines_from_desc


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

        self.assertEqual(expand(loop_seq, to_id), ["mda", ["d1"]])
        self.assertEqual(expand(ExecutionSequenceFactory.serial([]), to_id), [])
        self.assertEqual(expand(serial_seq, to_id), ["d1"])

        parallel_seq = ExecutionSequenceFactory.parallel([]).extend(d1)
        parallel_seq.extend(d2)

        loop_seq = ExecutionSequenceFactory.loop(mda, parallel_seq)
        self.assertEqual(expand(loop_seq, to_id), ["mda", [{"parallel": ["d1", "d2"]}]])

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
            design_space=SobieskiProblem().design_space,
            **options,
        )

        sc.formulation.minimize_objective = False
        return sc

    def test_xdsmize_mdf(self):
        """Test xdsmization of Sobieski problem solved with MDF with and without
        constraint."""
        scenario = self.build_mdo_scenario(inner_mda_name="MDAGaussSeidel")
        options = {
            "output_directory_path": ".",
            "html_output": False,
            "json_output": True,
            "outfilename": "xdsmized_sobieski_mdf.json",
        }
        assert_xdsm(scenario, **options)

        # with constraints
        options["outfilename"] = "xdsmized_sobieski_cstr_mdf.json"
        scenario.add_constraint(["g_1", "g_2", "g_3"], "ineq")
        assert_xdsm(scenario, **options)

        # without outdir
        xdsmizer = XDSMizer(scenario)
        xdsmizer.run(json_output=True, outfilename="xdsmized_sobieski_mdf.json")

        self.assertRaises(ValueError, xdsmizer._find_atom, Sellar1())

        xdsmizer.run(outfilename="xdsmized_sobieski_mdf.html")

    def test_xdsmize_idf(self):
        """Test xdsmization of Sobieski problem solved with IDF with and without
        constraint."""
        formulation = "IDF"
        scenario = self.build_mdo_scenario(formulation)
        scenario.xdsmize(html_output=False, json_output=True)
        options = {
            "output_directory_path": ".",
            "html_output": False,
            "json_output": True,
            "outfilename": "xdsmized_sobieski_idf.json",
        }
        assert_xdsm(scenario, **options)

        # with constraints
        options["outfilename"] = "xdsmized_sobieski_cstr_idf.json"
        for c_name in ["g_1", "g_2", "g_3"]:
            scenario.add_constraint(c_name, "ineq")
        assert_xdsm(scenario, **options)

    def test_xdsmize_bilevel(self):
        """Test xdsmization of Sobieski problem solved with bilevel."""
        design_space = SobieskiProblem().design_space
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

        sub_disciplines = [sc_prop, sc_aero, sc_str] + [mission]

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
        system_scenario.xdsmize(json_output=True)
        options = {
            "output_directory_path": ".",
            "latex_output": False,
            "html_output": False,
            "json_output": True,
            "outfilename": "xdsmized_sobieski_bilevel.json",
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

    deep_chain = MDOChain(
        [
            elementary_discipline(get_name(1), get_name(2)),
            elementary_discipline(get_name(2), get_name(3)),
        ]
    )

    inter_chain = MDOChain(
        [deep_chain, elementary_discipline(get_name(3), get_name(4))]
    )

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
        "html_output": False,
        "json_output": True,
        "outfilename": "xdsmized_nested_chains.json",
    }

    assert_xdsm(nested_chains, **options)


def test_xdsmize_nested_parallel_chain(tmp_wd, elementary_discipline):
    """Test the XDSM representation of nested ``MDOParallelChain``s.

    Here, we build a 3-levels nested chain.
    """

    def get_name(x: int) -> str:
        return f"x_{x}"

    beg_chain = elementary_discipline(get_name(1), get_name(2))

    deep_chain = MDOParallelChain(
        [
            elementary_discipline(get_name(2), get_name(3)),
            elementary_discipline(get_name(2), get_name(4)),
        ]
    )

    inter_chain = MDOParallelChain(
        [deep_chain, elementary_discipline(get_name(4), get_name(5))]
    )

    design_space = DesignSpace()
    design_space.add_variable(get_name(1))

    nested_chains = MDOScenario(
        [beg_chain, inter_chain],
        formulation="DisciplinaryOpt",
        objective_name=get_name(5),
        design_space=design_space,
    )

    options = {
        "html_output": False,
        "json_output": True,
        "outfilename": "xdsmized_nested_parallel_chains.json",
    }

    assert_xdsm(nested_chains, **options)


def assert_xdsm(scenario: MDOScenario, **options: Mapping[str, Any]) -> None:
    """Build and check the XDSM representation generated from a scenario.

    Args:
        scenario: The scenario from which the XDSM is generated.
        **options: The options for the XDSMizer.
    """
    fname = options["outfilename"]
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
    ref_filepath = current_dir / "data" / ref_file

    assert ref_filepath.exists(), (
        f"Reference {str(ref_filepath)} not found in data " f"directory."
    )

    with open(ref_filepath) as ref_file:
        xdsm_str = ref_file.read()
    expected = json.loads(xdsm_str)

    new_filepath = Path(generated_file)
    assert new_filepath.exists(), f"Generated {str(new_filepath)} not found."

    with open(new_filepath) as new_file:
        xdsm_str = new_file.read()
    xdsm_json = json.loads(xdsm_str)

    assert_xdsm_equal(expected, xdsm_json)


def assert_xdsm_equal(expected: XdsmType, generated: XdsmType) -> None:
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
        assert found, f"Node {str(expected_node)} not found."

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
        assert found, f"Edge {str(expected_edge)} not found."
    assert expected["workflow"] == generated["workflow"]


def test_xdsmize_mdf_mdoparallelchain(tmp_wd):
    """Test the XDSM representation of an MDF including an MDOParallelChain.

    In this case, the two MDAGaussSeidel created in the MDAChain must be parallel
    """
    disciplines = analytic_disciplines_from_desc(
        (
            {"a": "x"},
            {"y1": "x1", "b": "a+1"},
            {"x1": "1.-0.3*y1"},
            {"y2": "x2", "c": "a+2"},
            {"x2": "1.-0.3*y2"},
        )
    )
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
        "html_output": False,
        "json_output": True,
        "outfilename": "xdsmized_mdf_mdoparallelchain.json",
    }

    assert_xdsm(scenario, **options)
