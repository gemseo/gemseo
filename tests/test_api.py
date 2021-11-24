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
from __future__ import division, unicode_literals

import json
import re
from copy import deepcopy
from unittest import mock

import pytest
from numpy import arange, array, cos, linspace
from numpy import pi as np_pi
from numpy import sin
from six import string_types

from gemseo.api import (
    compute_doe,
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
    get_scenario_differentiation_modes,
    get_scenario_inputs_schema,
    get_scenario_options_schema,
    get_surrogate_options_schema,
    import_discipline,
    load_dataset,
    monitor_scenario,
    print_configuration,
)
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.problems.sobieski.wrappers import SobieskiMission
from gemseo.utils.py23_compat import Path


class Observer(object):
    def __init__(self):
        self.status_changes = 0

    def update(self, atom):
        self.status_changes += 1


def test_generate_n2_plot(tmp_wd):
    """Test the n2 plot with the Sobieski problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
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
    assert Path(file_path).exists()


def test_generate_coupling_graph(tmp_wd):
    """Test the coupling graph with the Sobieski problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    # TODO: reuse data and checks from test_dependency_graph
    disciplines = create_discipline(
        [
            "SobieskiMission",
            "SobieskiAerodynamics",
            "SobieskiStructure",
            "SobieskiPropulsion",
        ]
    )
    file_path = "coupl.pdf"
    generate_coupling_graph(disciplines, file_path)
    assert Path(file_path).exists()
    assert Path("coupl.dot").exists()


def test_get_algorithm_options_schema(tmp_wd):
    """Test that all available options are printed.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    schema_dict = get_algorithm_options_schema("SLSQP")
    assert "properties" in schema_dict
    assert len(schema_dict["properties"]) == 10

    schema_json = get_algorithm_options_schema("SLSQP", output_json=True)
    out_dict = json.loads(schema_json)
    for key, val in schema_dict.items():
        assert key in out_dict
        assert out_dict[key] == val

    with pytest.raises(ValueError, match="Algorithm named unknown is not available."):
        get_algorithm_options_schema("unknown")

    get_algorithm_options_schema("SLSQP", pretty_print=True)


def test_get_surrogate_options_schema(tmp_wd):
    """Test that the surrogate options schema is printed.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    get_surrogate_options_schema("RBFRegression")
    get_surrogate_options_schema("RBFRegression", pretty_print=True)


def test_create_scenario_and_monitor(tmp_wd):
    """Test the creation of a scenario from the SobieskiMission discipline.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    create_scenario(
        create_discipline("SobieskiMission"),
        "DisciplinaryOpt",
        "y_4",
        SobieskiProblem().read_design_space(),
    )

    with pytest.raises(
        ValueError, match="Unknown scenario type: unknown, use one of : 'MDO' or 'DOE'."
    ):
        create_scenario(
            create_discipline("SobieskiMission"),
            "DisciplinaryOpt",
            "y_4",
            SobieskiProblem().read_design_space(),
            scenario_type="unknown",
        )


def test_monitor_scenario(tmp_wd):
    """Test the scenario monitoring API method.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
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


def test_execute_post(tmp_wd):
    """Test the API method to call the post-processing factory.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    scenario = create_scenario(
        create_discipline("SobieskiMission"),
        "DisciplinaryOpt",
        "y_4",
        SobieskiProblem().read_design_space(),
    )
    scenario.execute({"algo": "SLSQP", "max_iter": 10})

    execute_post(scenario, "OptHistoryView", save=True, show=False)
    with pytest.raises(
        TypeError, match="Cannot post process type: {}".format(type(1234))
    ):
        execute_post(1234, "OptHistoryView")


def test_create_doe_scenario(tmp_wd):
    """Test the creation of a DOE scenario.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    create_scenario(
        create_discipline("SobieskiMission"),
        "DisciplinaryOpt",
        "y_4",
        SobieskiProblem().read_design_space(),
        scenario_type="DOE",
    )


def test_get_all_inputs(tmp_wd):
    """Check that all the inputs from the Sobieski problem are recovered.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    inputs = get_all_inputs(
        create_discipline(["SobieskiMission", "SobieskiAerodynamics"])
    )
    assert sorted(inputs) == sorted(
        ["y_12", "x_shared", "y_14", "x_2", "y_24", "y_32", "y_34"]
    )


def test_get_formulation_sub_options_schema(tmp_wd):
    """Check that the sub options schema is recovered for different formulations.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    sub_opts_schema = get_formulation_sub_options_schema(
        "MDF", main_mda_class="MDAJacobi"
    )
    props = sub_opts_schema["properties"]
    assert "acceleration" in props

    for formulation in get_available_formulations():
        if formulation == "MDF":
            opts = {"main_mda_class": "MDAJacobi"}
        elif formulation == "BiLevel" or formulation == "BLISS98B":
            opts = {"mda_name": "MDAGaussSeidel"}
        else:
            opts = {}
        get_formulation_sub_options_schema(formulation, **opts)
        get_formulation_sub_options_schema(formulation, pretty_print=True, **opts)


def test_get_all_in_out_puts_recursive(tmp_wd):
    """Test the recursive option when retrieving all inputs/outputs.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
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

    inputs = get_all_inputs([sc_aero, sc_struct, miss], recursive=True)
    assert sorted(inputs) == sorted(
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


def test_get_scenario_inputs_schema(tmp_wd):
    """Check that the scenario inputs schema is retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    aero = create_discipline(["SobieskiAerodynamics"])
    design_space = SobieskiProblem().read_design_space()
    sc_aero = create_scenario(
        aero, "DisciplinaryOpt", "y_24", design_space.filter("x_2")
    )

    schema = get_scenario_inputs_schema(sc_aero)
    assert "algo_options" in schema["properties"]
    assert "algo" in schema["properties"]

    get_scenario_inputs_schema(sc_aero, pretty_print=True)


def test_exec_algo(tmp_wd):
    """Test the execution of an algorithm with the Rosenbrock problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    problem = Rosenbrock()
    sol = execute_algo(problem, "L-BFGS-B", max_iter=200)
    assert abs(sol.f_opt) < 1e-8

    sol = execute_algo(problem, "lhs", algo_type="doe", n_samples=200)
    assert abs(sol.f_opt) < 1e-8

    with pytest.raises(
        ValueError,
        match="Unknown algo type: unknown_algo, please use 'doe' or 'opt' !",
    ):
        execute_algo(problem, "lhs", "unknown_algo", n_samples=200)


def test_get_scenario_options_schema(tmp_wd):
    """Check that the scenario options schema is retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    schema = get_scenario_options_schema("MDO")
    assert "name" in schema["properties"]

    with pytest.raises(ValueError, match="Unknown scenario type UnknownType"):
        get_scenario_options_schema("UnknownType")

    get_scenario_options_schema("MDO", pretty_print=True)


def test_get_mda_options_schema(tmp_wd):
    """Check that the mda options schema are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    schema = get_mda_options_schema("MDAJacobi")
    assert "name" in schema["properties"]

    get_mda_options_schema("MDAJacobi", pretty_print=True)


def test_get_all_outputs(tmp_wd):
    """Test that all discipline outputs are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    outs = get_all_outputs(
        create_discipline(["SobieskiMission", "SobieskiAerodynamics"])
    )
    assert sorted(outs) == sorted(["y_23", "y_24", "y_4", "y_2", "g_2", "y_21"])


def test_get_available_opt_algorithms(tmp_wd):
    """Check that the optimization algorithms are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    algos = get_available_opt_algorithms()
    assert "SLSQP" in algos
    assert "L-BFGS-B" in algos
    assert "TNC" in algos


def test_get_available_doe_algorithms(tmp_wd):
    """Test that the doe algorithms are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    algos = get_available_doe_algorithms()
    assert "lhs" in algos
    assert "fullfact" in algos


def test_get_available_formulations(tmp_wd):
    """Test that the available formulations are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    formulations = get_available_formulations()
    assert "MDF" in formulations
    assert "DisciplinaryOpt" in formulations
    assert "IDF" in formulations


def test_get_available_post_processings(tmp_wd):
    """Test that the available post-processing methods are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    post_processors = get_available_post_processings()
    assert "OptHistoryView" in post_processors
    assert "RadarChart" in post_processors


def test_get_available_surrogates(tmp_wd):
    """Test that the available surrogates are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    surrogates = get_available_surrogates()
    assert "RBFRegression" in surrogates
    assert "LinearRegression" in surrogates


def test_get_available_disciplines(tmp_wd):
    """Test that the available disciplines are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    disciplines = get_available_disciplines()
    assert "SobieskiMission" in disciplines
    assert "Sellar1" in disciplines


def test_create_discipline(tmp_wd):
    """Test that API method creates a discipline properly.

    Test exceptions when the options dictionary does not follow the
    specified json grammar.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
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

    with pytest.raises(InvalidDataException, match="Invalid data in"):
        create_discipline("SobieskiMission", **options_fail)


def test_create_surrogate(tmp_wd):
    """Test the creation of a surrogate discipline.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
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


def test_create_scalable(tmp_wd):
    """Test the creation of a scalable discipline.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """

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


def test_create_mda(tmp_wd):
    """Test the creation of an MDA from the Sobieski disciplines.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
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


def test_get_available_mdas(tmp_wd):
    """Test that the available MDA solvers are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    mdas = get_available_mdas()
    assert "MDAGaussSeidel" in mdas


def test_get_discipline_inputs_schema(tmp_wd):
    """Test that the discipline input schemas are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    mission = create_discipline("SobieskiMission")
    schema_dict = get_discipline_inputs_schema(mission, False)
    for key in mission.get_input_data_names():
        assert key in schema_dict["properties"]

    schema_str = get_discipline_inputs_schema(mission, True)
    assert isinstance(schema_str, string_types)
    get_discipline_inputs_schema(mission, False, pretty_print=True)


def test_get_discipline_outputs_schema(tmp_wd):
    """Test that the discipline output schemas are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    mission = create_discipline("SobieskiMission")
    schema_dict = get_discipline_outputs_schema(mission, False)
    for key in mission.get_output_data_names():
        assert key in schema_dict["properties"]

    schema_str = get_discipline_outputs_schema(mission, True)
    assert isinstance(schema_str, string_types)
    get_discipline_outputs_schema(mission, False, pretty_print=True)


def test_get_scenario_differenciation_modes(tmp_wd):
    """Test that the scenario differentiation modes are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    modes = get_scenario_differentiation_modes()
    for mode in modes:
        assert isinstance(mode, string_types)


def test_get_post_processing_options_schema(tmp_wd):
    """Test that the post-processing option schemas are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    for post in get_available_post_processings():
        get_post_processing_options_schema(post)


def test_get_formulation_options_schema(tmp_wd):
    """Test that the formulation options schemas are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
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
    get_formulation_options_schema("IDF", pretty_print=True)


def test_get_discipline_options_schema(tmp_wd):
    """Test that the discipline options schemas are retrived correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
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
        get_discipline_options_schema(disc, pretty_print=True)


def test_get_discipline_options_defaults(tmp_wd):
    """Test that the discipline options defaults are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    for disc in ["SobieskiMission", "MDOChain", "AnalyticDiscipline"]:
        defaults = get_discipline_options_defaults(disc)
        assert len(defaults) > 0


def test_get_default_sub_options_values(tmp_wd):
    """Test that the default sub options values are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """

    defaults = get_formulations_sub_options_defaults("MDF", main_mda_class="MDAChain")
    assert defaults is not None

    defaults = get_formulations_sub_options_defaults("DisciplinaryOpt")
    assert defaults is None


def test_get_formulations_options_defaults(tmp_wd):
    """Test that the formulation options defaults are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    for form in ["MDF", "BiLevel"]:
        defaults = get_formulations_options_defaults(form)
        assert len(defaults) > 0


def test_get_available_scenario_types(tmp_wd):
    """Test that the available scenario types are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    scen_types = get_available_scenario_types()
    assert "MDO" in scen_types
    assert "DOE" in scen_types


def test_create_parameter_space(tmp_wd):
    """Test the creation of a parameter space.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    parameter_space = create_parameter_space()
    parameter_space.add_variable("name", 1, var_type="float", l_b=-1, u_b=1, value=0)
    parameter_space.add_random_variable("other_name", "OTNormalDistribution")
    parameter_space.check()


def test_create_design_space(tmp_wd):
    """Test the creation of a design space.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    design_space = create_design_space()
    design_space.add_variable("name", 1, var_type="float", l_b=-1, u_b=1, value=0)
    design_space.check()


def test_export_design_space(tmp_wd):
    """Test that a design space can be exported to a text or h5 file.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    design_space = create_design_space()
    design_space.add_variable("name", 1, var_type="float", l_b=-1, u_b=1, value=0)
    export_design_space(design_space, "design_space.txt")
    export_design_space(design_space, "design_space.h5", True)


def test_create_cache(tmp_wd):
    """Test the creation of a cache.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    cache = create_cache("MemoryFullCache")
    assert cache.get_length() == 0


def test_get_available_caches(tmp_wd):
    """Test that the available caches are retrieved correctly.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    caches = get_available_caches()
    # plugins may add classes
    assert set(caches) <= set(["HDF5Cache", "MemoryFullCache", "SimpleCache"])


def test_load_dataset(tmp_wd):
    """Test the load_dataset method with the `BurgersDataset`.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    burgers = load_dataset("BurgersDataset")
    assert burgers.length == 30


def test_print_configuration(capfd):
    """Check that the |g| configuration is shown to the user.

    Args:
        capfd: Fixture capture outputs sent to `stdout` and
            `stderr`.
    """
    print_configuration()

    out, err = capfd.readouterr()
    assert not err

    gemseo_modules = [
        "MDODiscipline",
        "OptimizationLibrary",
        "DOELibrary",
        "MLRegressionAlgo",
        "MDOFormulation",
        "MDA",
        "OptPostProcessor",
    ]

    for module in gemseo_modules:
        header_patterns = (
            r"\+-+\+$\n"
            r"\|\s+{}\s+\|$\n"
            r"\+-+\+-+\+-+\+$\n"
            r"\|\s+Module\s+\|\s+Is available \?\s+\|\s+Purpose or error "
            r"message\s+\|$\n".format(module)
        )

        expected = re.compile(header_patterns, re.MULTILINE)
        assert bool(re.search(expected, out))


def test_get_schema_pretty_print(capfd):
    """Test that the post-processing options schemas are printed correctly.

    Args:
        capfd: Fixture capture outputs sent to `stdout` and
            `stderr`.
    """
    # A pattern for table headers.
    expected = re.compile(
        r"\+-+\+-+\+-+\+$\n\|\s+Name\s+\|\s+Description\s+\|\s+Type\s+\|$\n",
        re.MULTILINE,
    )

    for post in get_available_post_processings():
        get_post_processing_options_schema(post, pretty_print=True)

        out, err = capfd.readouterr()
        assert not err

        assert bool(re.search(expected, out))


@pytest.fixture(scope="module")
def variables_space():
    """A mock design space."""
    design_space = mock.Mock()
    design_space.dimension = 2
    design_space.untransform_vect = mock.Mock(return_value=arange(6).reshape((3, 2)))
    return design_space


def test_compute_doe_transformed(variables_space):
    """Check the computation of a transformed DOE in a variables space."""
    points = compute_doe(variables_space, size=3, algo_name="lhs", unit_sampling=True)
    assert points.shape == (3, 2)
    assert points.max() <= 1.0
    assert points.min() >= 0.0
    variables_space.untransform_vect.assert_not_called()


def test_compute_doe_nontransformed(variables_space):
    """Check the computation of a non-transformed DOE in a variables space."""
    points = compute_doe(variables_space, size=3, algo_name="lhs")
    assert points.shape == (3, 2)
    variables_space.untransform_vect.assert_called_once()


def test_import_discipline(tmp_wd):
    """Check that a discipline performs correctly after import."""
    file_path = tmp_wd / "saved_discipline.pkl"

    discipline = create_discipline("AnalyticDiscipline", expressions_dict={"y": "2*x"})
    discipline.execute()
    discipline.serialize(file_path)

    loaded_discipline = import_discipline(file_path)
    loaded_discipline.execute()

    assert loaded_discipline.local_data["x"] == discipline.local_data["x"]
    assert loaded_discipline.local_data["y"] == discipline.local_data["y"]
