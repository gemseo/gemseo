# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import json
import unittest
from copy import deepcopy
from os.path import abspath, dirname, exists, join

import pytest

from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.mdo_scenario import MDODiscipline, MDOScenario
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)
from gemseo.utils.xdsmizer import XDSMizer, expand


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
            design_space=SobieskiProblem().read_design_space(),
            **options
        )

        sc.formulation.minimize_objective = False
        return sc

    def test_xdsmize_mdf(self):
        """Test xdsmization of Sobieski problem solved with MDF with and without
        constraint."""
        scenario = self.build_mdo_scenario("MDF", sub_mda_class="MDAGaussSeidel")
        options = {
            "output_directory_path": ".",
            "html_output": False,
            "json_output": True,
            "outfilename": "xdsmized_sobieski_mdf.json",
        }
        self._assert_xdsm(scenario, **options)

        # with constraints
        options["outfilename"] = "xdsmized_sobieski_cstr_mdf.json"
        scenario.add_constraint(["g_1", "g_2", "g_3"], "ineq")
        self._assert_xdsm(scenario, **options)

        # without outdir
        xdsmizer = XDSMizer(scenario)
        xdsmizer.run(
            html_output=True,
            json_output=True,
            open_browser=False,
            outfilename="xdsmized_sobieski_mdf.json",
        )

        self.assertRaises(ValueError, xdsmizer._find_atom, Sellar1())

        xdsmizer.run(
            html_output=True,
            json_output=False,
            outfilename="xdsmized_sobieski_mdf.html",
            open_browser=False,
        )

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
        self._assert_xdsm(scenario, **options)

        # with constraints
        options["outfilename"] = "xdsmized_sobieski_cstr_idf.json"
        for c_name in ["g_1", "g_2", "g_3"]:
            scenario.add_constraint(c_name, "ineq")
        self._assert_xdsm(scenario, **options)

    def test_xdsmize_bilevel(self):
        """Test xdsmization of Sobieski problem solved with bilevel."""

        design_space = SobieskiProblem().read_design_space()
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
        system_scenario.xdsmize(html_output=True, json_output=True, open_browser=False)
        options = {
            "output_directory_path": ".",
            "latex_output": False,
            "html_output": False,
            "json_output": True,
            "outfilename": "xdsmized_sobieski_bilevel.json",
        }
        self._assert_xdsm(system_scenario, **options)

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

    #         system_scenario_par.xdsmize()

    def _assert_xdsm(self, scenario, **options):
        """Assert XDSM equality taking file 'ref_<fname> as reference.

        :param scenario: the scenario to be xdsmize
        @options options for xdsmise function
        :param **options:
        """
        fname = options["outfilename"]
        xdsmizer = XDSMizer(scenario)
        # xdsm_json = xdsmizer.run(**options)
        # self._assert_xdsm_json(fname, xdsm_json)
        xdsmizer.run(**options)
        self._assert_xdsm_file_ok(fname)

    def _assert_xdsm_file_ok(self, fname):
        """Tests XDSM equality taking file 'ref_<fname> as reference and new generated
        file <fname>

        :param fname: filename containing generated XDSM
        """
        ref_filepath = join(dirname(abspath(__file__)), "data", fname)
        # Erase reference files
        #         import shutil
        #         shutil.copyfile(fname, ref_filepath)
        self.assertTrue(exists(ref_filepath), ref_filepath + " not found!")
        with open(ref_filepath, "r") as ref_file:
            xdsm_str = ref_file.read()
        expected = json.loads(xdsm_str)

        new_filepath = fname
        assert exists(new_filepath)
        with open(new_filepath, "r") as new_file:
            xdsm_str = new_file.read()
        xdsm_json = json.loads(xdsm_str)
        self._assert_xdsm_equal(expected, xdsm_json)

    def _assert_xdsm_equal(self, expected, xdsm_json):
        """

        :param expected: param xdsm_json:
        :param xdsm_json:

        """
        self.assertEqual(sorted(expected.keys()), sorted(xdsm_json.keys()))
        for key in expected:
            self._assert_level_xdsm_equal(expected[key], xdsm_json[key])

    def _assert_level_xdsm_equal(self, expected, xdsm_json):
        """

        :param expected: param xdsm_json:
        :param xdsm_json:

        """
        self.assertEqual(len(expected["nodes"]), len(xdsm_json["nodes"]))
        for expected_node in expected["nodes"]:
            found = False
            for node in xdsm_json["nodes"]:
                if node["id"] == expected_node["id"]:
                    self.assertEqual(expected_node["name"], node["name"])
                    self.assertEqual(expected_node["type"], node["type"])
                    found = True
            self.assertTrue(found, "Node " + str(expected_node) + " not found")
        for expected_edge in expected["edges"]:
            found = False
            for edge in xdsm_json["edges"]:
                if (
                    edge["from"] == expected_edge["from"]
                    and edge["to"] == expected_edge["to"]
                ):
                    self.assertSetEqual(
                        set(expected_edge["name"].split(", ")),
                        set(edge["name"].split(", ")),
                    )
                    found = True
            self.assertTrue(found, "Edge " + str(expected_edge) + " not found")
        self.assertListEqual(expected["workflow"], xdsm_json["workflow"])
