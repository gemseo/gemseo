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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import absolute_import, division, print_function, unicode_literals

import json
import unittest
from copy import deepcopy
from os.path import exists

import pytest
from numpy import array, cos, linspace
from numpy import pi as np_pi
from numpy import sin
from six import string_types

from gemseo.api import (
    create_cache,
    create_design_space,
    create_discipline,
    create_mda,
    create_parameter_space,
    create_scalable,
    create_scenario,
    create_surrogate,
    execute_algo,
    execute_post,
    export_design_space,
    generate_coupling_graph,
    generate_n2_plot,
    get_algorithm_options_schema,
    get_all_inputs,
    get_all_outputs,
    get_available_caches,
    get_available_disciplines,
    get_available_doe_algorithms,
    get_available_formulations,
    get_available_mdas,
    get_available_opt_algorithms,
    get_available_post_processings,
    get_available_scenario_types,
    get_available_surrogates,
    get_discipline_inputs_schema,
    get_discipline_options_defaults,
    get_discipline_options_schema,
    get_discipline_outputs_schema,
    get_formulation_options_schema,
    get_formulation_sub_options_schema,
    get_formulations_options_defaults,
    get_formulations_sub_options_defaults,
    get_mda_options_schema,
    get_post_processing_options_schema,
    get_scenario_differenciation_modes,
    get_scenario_inputs_schema,
    get_scenario_options_schema,
    get_surrogate_options_schema,
    load_dataset,
    monitor_scenario,
    print_configuration,
)
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.grammar import InvalidDataException
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import SobieskiMission


class Observer(object):
    def __init__(self):
        self.status_changes = 0

    def update(self, atom):
        self.status_changes += 1


@pytest.mark.usefixtures("tmp_wd")
class TestAPI(unittest.TestCase):
    def test_generate_n2_plot(self):
        disciplines = create_discipline(
            [
                "SobieskiMission",
                "SobieskiAerodynamics",
                "SobieskiStructure",
                "SobieskiPropulsion",
            ]
        )
        file_path = "n2.png"
        generate_n2_plot(disciplines, file_path, save=True, show=False, figsize=(5, 5))
        assert exists(file_path)

    def test_generate_coupling_graph(self):
        disciplines = create_discipline(
            [
                "SobieskiMission",
                "SobieskiAerodynamics",
                "SobieskiStructure",
                "SobieskiPropulsion",
            ]
        )
        base_path = "coupl"
        gv_path = "coupl.gv"
        file_path = base_path + ".pdf"
        generate_coupling_graph(disciplines, file_path)
        assert exists(file_path)
        assert exists(gv_path)

    def test_get_algorithm_options_schema(self):
        schema_dict = get_algorithm_options_schema("SLSQP")
        assert "properties" in schema_dict
        assert len(schema_dict["properties"]) == 10

        schema_json = get_algorithm_options_schema("SLSQP", output_json=True)
        out_dict = json.loads(schema_json)
        for key, val in schema_dict.items():
            assert key in out_dict
            assert out_dict[key] == val

        self.assertRaises(ValueError, get_algorithm_options_schema, "unknown")

        schema_dict = get_algorithm_options_schema("SLSQP", pretty_print=True)

    def test_get_surrogate_options_schema(self):
        get_surrogate_options_schema("RBFRegression")
        get_surrogate_options_schema("RBFRegression", pretty_print=True)

    def test_create_scenario_and_monitor(self):
        scenario = create_scenario(
            create_discipline("SobieskiMission"),
            "DisciplinaryOpt",
            "y_4",
            SobieskiProblem().read_design_space(),
        )

        observer = Observer()
        monitor_scenario(scenario, observer)

        scenario.execute({"algo": "SLSQP", "max_iter": 10})
        assert (
            observer.status_changes
            >= 2 * scenario.formulation.opt_problem.objective.n_calls
        )
        execute_post(scenario, "OptHistoryView", save=True, show=False)
        self.assertRaises(
            ValueError,
            create_scenario,
            create_discipline("SobieskiMission"),
            "DisciplinaryOpt",
            "y_4",
            SobieskiProblem().read_design_space(),
            scenario_type="unknown",
        )
        self.assertRaises(TypeError, execute_post, 1234, "OptHistoryView")

    def test_create_doe_scenario(self):
        create_scenario(
            create_discipline("SobieskiMission"),
            "DisciplinaryOpt",
            "y_4",
            SobieskiProblem().read_design_space(),
            scenario_type="DOE",
        )

    def test_get_all_inputs(self):
        inpts = get_all_inputs(
            create_discipline(["SobieskiMission", "SobieskiAerodynamics"])
        )
        assert sorted(inpts) == sorted(
            ["y_12", "x_shared", "y_14", "x_2", "y_24", "y_32", "y_34"]
        )

    def test_get_formulation_sub_options_schema(self):
        sub_opts_schema = get_formulation_sub_options_schema(
            "MDF", main_mda_class="MDAJacobi"
        )
        props = sub_opts_schema["properties"]
        assert "acceleration" in props

        for formu in get_available_formulations():
            if formu == "MDF":
                opts = {"main_mda_class": "MDAJacobi"}
            elif formu == "BiLevel" or formu == "BLISS98B":
                opts = {"mda_name": "MDAGaussSeidel"}
            else:
                opts = {}
            sub_opts_schema = get_formulation_sub_options_schema(formu, **opts)
            sub_opts_schema = get_formulation_sub_options_schema(
                formu, pretty_print=True, **opts
            )

    def test_get_all_in_out_puts_recursive(self):
        miss, aero, struct = create_discipline(
            ["SobieskiMission", "SobieskiAerodynamics", "SobieskiStructure"]
        )
        design_space = SobieskiProblem().read_design_space()
        sc_aero = create_scenario(
            aero, "DisciplinaryOpt", "y_24", deepcopy(design_space).filter("x_2")
        )

        sc_struct = create_scenario(
            struct, "DisciplinaryOpt", "y_14", deepcopy(design_space).filter("x_1")
        )

        inpts = get_all_inputs([sc_aero, sc_struct, miss], recursive=True)
        assert sorted(inpts) == sorted(
            [
                "x_1",
                "x_2",
                "x_shared",
                "y_12",
                "y_14",
                "y_21",
                "y_24",
                "y_31",
                "y_32",
                "y_34",
            ]
        )

        get_all_outputs([sc_aero, sc_struct, miss], recursive=True)

    def test_get_scenario_inputs_schema(self):
        aero = create_discipline(["SobieskiAerodynamics"])
        design_space = SobieskiProblem().read_design_space()
        sc_aero = create_scenario(
            aero, "DisciplinaryOpt", "y_24", design_space.filter("x_2")
        )

        schema = get_scenario_inputs_schema(sc_aero)
        assert "algo_options" in schema["properties"]
        assert "algo" in schema["properties"]
        schema = get_scenario_inputs_schema(sc_aero, pretty_print=True)

    def test_exec_algo(self):
        problem = Rosenbrock()
        sol = execute_algo(problem, "L-BFGS-B", max_iter=200)
        assert abs(sol.f_opt) < 1e-8
        sol = execute_algo(problem, "lhs", algo_type="doe", n_samples=200)
        assert abs(sol.f_opt) < 1e-8
        self.assertRaises(
            ValueError, execute_algo, problem, "lhs", "unknown_algo", n_samples=200
        )

    def test_get_scenario_options_schema(self):
        schema = get_scenario_options_schema("MDO")
        assert "name" in schema["properties"]
        self.assertRaises(ValueError, get_scenario_options_schema, "UnkwnonType")
        schema = get_scenario_options_schema("MDO", pretty_print=True)

    def test_get_mda_options_schema(self):
        schema = get_mda_options_schema("MDAJacobi")
        assert "name" in schema["properties"]
        schema = get_mda_options_schema("MDAJacobi", pretty_print=True)

    def test_get_all_outputs(self):
        outs = get_all_outputs(
            create_discipline(["SobieskiMission", "SobieskiAerodynamics"])
        )
        assert sorted(outs) == sorted(["y_23", "y_24", "y_4", "y_2", "g_2", "y_21"])

    def test_get_available_opt_algorithms(self):
        algos = get_available_opt_algorithms()
        assert "SLSQP" in algos
        assert "L-BFGS-B" in algos
        assert "TNC" in algos

    def test_get_available_doe_algorithms(self):
        algos = get_available_doe_algorithms()
        assert "lhs" in algos
        assert "fullfact" in algos

    def test_get_available_formulations(self):
        formulations = get_available_formulations()
        assert "MDF" in formulations
        assert "DisciplinaryOpt" in formulations
        assert "IDF" in formulations

    def test_get_available_post_processings(self):
        visus = get_available_post_processings()
        assert "OptHistoryView" in visus
        assert "RadarChart" in visus

    def test_get_available_surrogates(self):
        surrogates = get_available_surrogates()
        assert "RBFRegression" in surrogates
        assert "LinearRegression" in surrogates

    def test_get_available_disciplines(self):
        disciplines = get_available_disciplines()
        assert "SobieskiMission" in disciplines
        assert "Sellar1" in disciplines

    def test_create_discipline(self):
        options = {
            "dtype": "float64",
            "linearization_mode": "auto",
            "cache_type": "SimpleCache",
        }
        miss = create_discipline("SobieskiMission", **options)
        miss.execute()
        assert isinstance(miss, MDODiscipline)

        options_fail = {
            "dtype": "float64",
            "linearization_mode": "finite_differences",
            "cache_type": MDODiscipline.SIMPLE_CACHE,
        }
        self.assertRaises(
            InvalidDataException, create_discipline, "SobieskiMission", **options_fail
        )

    def test_create_surrogate(self):
        disc = SobieskiMission()
        input_names = ["y_24", "y_34"]
        disc.set_cache_policy(disc.MEMORY_FULL_CACHE)
        design_space = SobieskiProblem().read_design_space()
        design_space.filter(input_names)
        doe = DOEScenario([disc], "DisciplinaryOpt", "y_4", design_space)
        doe.execute({"algo": "fullfact", "n_samples": 10})
        surr = create_surrogate(
            "RBFRegression",
            disc.cache.export_to_dataset(),
            input_names=["y_24", "y_34"],
        )
        outs = surr.execute({"y_24": array([1.0]), "y_34": array([1.0])})

        assert outs["y_4"] > 0.0

    def test_create_scalable(self):
        def f_1(x_1, x_2, x_3):
            return sin(2 * np_pi * x_1) + cos(2 * np_pi * x_2) + x_3

        def f_2(x_1, x_2, x_3):
            return sin(2 * np_pi * x_1) * cos(2 * np_pi * x_2) - x_3

        data = Dataset("sinus")
        x1_val = x2_val = x3_val = linspace(0.0, 1.0, 10)[:, None]
        data.add_variable("x1", x1_val, data.INPUT_GROUP, True)
        data.add_variable("x2", x2_val, data.INPUT_GROUP, True)
        data.add_variable("x3", x2_val, data.INPUT_GROUP, True)
        data.add_variable("y1", f_1(x1_val, x2_val, x3_val), data.OUTPUT_GROUP, True)
        data.add_variable("y2", f_2(x1_val, x2_val, x3_val), data.OUTPUT_GROUP, True)
        create_scalable("ScalableDiagonalModel", data, fill_factor=0.7)

    def test_create_mda(self):
        disciplines = create_discipline(
            [
                "SobieskiAerodynamics",
                "SobieskiPropulsion",
                "SobieskiStructure",
                "SobieskiMission",
            ]
        )
        mda = create_mda("MDAGaussSeidel", disciplines)
        mda.execute()
        assert mda.residual_history[-1][0] < 1e-4

    def test_get_available_mdas(self):
        mdas = get_available_mdas()
        assert "MDAGaussSeidel" in mdas

    def test_print_configuration(self):
        print_configuration()

    def test_get_discipline_inputs_schema(self):
        mission = create_discipline("SobieskiMission")
        schema_dict = get_discipline_inputs_schema(mission, False)
        for key in mission.get_input_data_names():
            assert key in schema_dict["properties"]

        schema_str = get_discipline_inputs_schema(mission, True)
        assert isinstance(schema_str, string_types)
        schema_str = get_discipline_inputs_schema(mission, False, pretty_print=True)

    def test_get_discipline_outputs_schema(self):
        mission = create_discipline("SobieskiMission")
        schema_dict = get_discipline_outputs_schema(mission, False)
        for key in mission.get_output_data_names():
            assert key in schema_dict["properties"]

        schema_str = get_discipline_outputs_schema(mission, True)
        assert isinstance(schema_str, string_types)
        schema_str = get_discipline_outputs_schema(mission, False, pretty_print=True)

    def test_get_scenario_differenciation_modes(self):
        modes = get_scenario_differenciation_modes()
        for mode in modes:
            assert isinstance(mode, string_types)

    def test_get_post_processing_options_schema(self):
        for post in get_available_post_processings():
            schema = get_post_processing_options_schema(post)
            if post != "KMeans":
                assert "show" in schema["properties"]
            else:
                assert "n_clusters" in schema["properties"]
            schema = get_post_processing_options_schema(post, pretty_print=True)

    def test_get_formulation_options_schema(self):
        mdf_schema = get_formulation_options_schema("MDF")
        for prop in ["maximize_objective", "sub_mda_class"]:
            assert prop in mdf_schema["required"]

        idf_schema = get_formulation_options_schema("IDF")
        for prop in [
            "maximize_objective",
            "normalize_constraints",
            "parallel_exec",
            "use_threading",
        ]:
            assert prop in idf_schema["required"]
        idf_schema = get_formulation_options_schema("IDF", pretty_print=True)

    def test_get_discipline_options_schema(self):
        for disc in ["SobieskiMission", "MDOChain", "AnalyticDiscipline"]:
            schema = get_discipline_options_schema(disc)
            props = schema["properties"]

            for opts in [
                "jac_approx_type",
                "linearization_mode",
                "cache_hdf_file",
                "cache_hdf_node_name",
            ]:
                assert opts in props
            schema = get_discipline_options_schema(disc, pretty_print=True)

    def test_get_discipline_options_defaults(self):
        for disc in ["SobieskiMission", "MDOChain", "AnalyticDiscipline"]:
            defaults = get_discipline_options_defaults(disc)
            assert len(defaults) > 0

    def test_get_default_sub_options_values(self):

        defaults = get_formulations_sub_options_defaults(
            "MDF", main_mda_class="MDAChain"
        )
        assert defaults is not None

        defaults = get_formulations_sub_options_defaults("DisciplinaryOpt")
        assert defaults is None

    def test_get_formulations_options_defaults(self):
        for form in ["MDF", "BiLevel"]:
            defaults = get_formulations_options_defaults(form)
            assert len(defaults) > 0

    def test_get_available_scenario_types(self):
        scen_types = get_available_scenario_types()
        assert "MDO" in scen_types
        assert "DOE" in scen_types

    def test_create_parameter_space(self):
        parameter_space = create_parameter_space()
        parameter_space.add_variable(
            "name", 1, var_type="float", l_b=-1, u_b=1, value=0
        )
        parameter_space.add_random_variable("other_name", "OTNormalDistribution")
        parameter_space.check()

    def test_create_design_space(self):
        design_space = create_design_space()
        design_space.add_variable("name", 1, var_type="float", l_b=-1, u_b=1, value=0)
        design_space.check()

    def test_export_design_space(self):
        design_space = create_design_space()
        design_space.add_variable("name", 1, var_type="float", l_b=-1, u_b=1, value=0)
        export_design_space(design_space, "design_space.txt")
        export_design_space(design_space, "design_space.h5", True)

    def test_create_cache(self):
        cache = create_cache("MemoryFullCache")
        assert cache.get_length() == 0

    def test_get_available_caches(self):
        caches = get_available_caches()
        assert "AbstractFullCache" in caches

    def test_load_dataset(self):
        burgers = load_dataset("BurgersDataset")
        assert burgers.length == 30
