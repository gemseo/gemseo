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

import numpy as np
import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.core.mdofunctions.consistency_constraint import ConsistencyCstr
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.idf import IDF
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure

from .formulations_basetest import FakeDiscipline


def test_build_func_from_disc():
    """"""
    pb = SobieskiProblem("complex128")
    disciplines = [
        SobieskiMission("complex128"),
        SobieskiAerodynamics("complex128"),
        SobieskiPropulsion("complex128"),
        SobieskiStructure("complex128"),
    ]
    idf = IDF(disciplines, "y_4", pb.design_space)
    x_names = idf.get_optim_variables_names()
    x_dict = pb.get_default_inputs(x_names)
    x_vect = np.concatenate([x_dict[k] for k in x_names])

    for c_name in ["g_1", "g_2", "g_3"]:
        idf.add_constraint(c_name, constraint_type="ineq")
    opt = idf.opt_problem
    opt.objective.check_grad(x_vect, "ComplexStep", 1e-30, error_max=1e-4)
    for cst in opt.constraints:
        cst.check_grad(x_vect, "ComplexStep", 1e-30, error_max=1e-4)

    for func_name in list(pb.get_default_inputs().keys()):
        if func_name.startswith("Y"):
            func = idf._build_func_from_outputs([func_name])
            func.check_grad(x_vect, "ComplexStep", 1e-30, error_max=1e-4)

    for coupl in idf.coupling_structure.strong_couplings:
        func = ConsistencyCstr([coupl], idf)
        func.check_grad(x_vect, "ComplexStep", 1e-30, error_max=1e-4)


@pytest.mark.parametrize(
    "options, expected_feasible",
    [
        (
            {
                "linearize": False,
                "dtype": "complex128",
                "normalize_cstr": True,
                "eq_tolerance": 1e-4,
                "ineq_tolerance": 1e-4,
                "n_processes": 1,
                "max_iter": 50,
            },
            True,
        ),
        (
            {
                "linearize": True,
                "dtype": "float64",
                "normalize_cstr": False,
                "n_processes": 2,
                "max_iter": 50,
            },
            False,
        ),
        (
            {
                "linearize": True,
                "dtype": "float64",
                "normalize_cstr": True,
                "n_processes": 1,
                "max_iter": 50,
            },
            True,
        ),
        (
            {
                "linearize": True,
                "dtype": "float64",
                "normalize_cstr": True,
                "n_processes": 2,
                "use_threading": True,
                "max_iter": 50,
            },
            True,
        ),
        (
            {
                "linearize": True,
                "dtype": "float64",
                "normalize_cstr": True,
                "n_processes": 2,
                "use_threading": False,
                "max_iter": 2,
            },
            False,
        ),
    ],
)
def test_idf_execution(
    options,
    expected_feasible,
    generate_idf_scenario,
    caplog,
):
    """Test the IDF formulation with an :class:`.MDOScenario`.

    Args:
        options: The options for the generate_idf_scenario fixture.
        expected_feasible: Whether the optimization result is expected to be feasible.
        generate_idf_scenario: Fixture that returns an :class:`.MDOScenario` with an IDF
            formulation with custom arguments.
        caplog: Fixture to access and control log capturing.
    """
    obj_opt, is_feasible = generate_idf_scenario(
        "SLSQP",
        **options,
    )

    if options["max_iter"] == 50:
        assert 3962.0 < obj_opt < 3965.0

    assert is_feasible == expected_feasible

    if options["n_processes"] > 1:
        assert "Running IDF formulation in parallel on n_processes" in caplog.text


def test_fail_idf_no_coupl(generate_idf_scenario):
    """Test an exception when the coupling variables are not in the Design Space.

    Args:
        generate_idf_scenario: Fixture that returns an :class:`.MDOScenario` with an IDF
            formulation with custom arguments.
    """
    with pytest.raises(
        ValueError,
        match="IDF formulation needs coupling variables as design variables, "
        r"missing variables: \{.*\}\.",
    ):
        generate_idf_scenario(
            "SLSQP",
            linearize=False,
            dtype="float64",
            normalize_cstr=True,
            remove_coupl_from_ds=True,
        )


def test_expected_workflow():
    """"""
    disc1 = FakeDiscipline("d1")
    disc2 = FakeDiscipline("d2")
    disc3 = FakeDiscipline("d3")
    idf = IDF([disc1, disc2, disc3], "d3_y", DesignSpace())
    expected = "(d1(None), d2(None), d3(None), )"
    assert str(idf.get_expected_workflow()) == expected


def test_expected_dataflow():
    """"""
    disc1 = FakeDiscipline("d1")
    disc2 = FakeDiscipline("d2")
    disc3 = FakeDiscipline("d3")
    idf = IDF([disc1, disc2, disc3], "d3_y", DesignSpace())
    assert idf.get_expected_dataflow() == []


def test_idf_start_equilibrium():
    """Initial value of coupling variables set at equilibrium."""
    disciplines = [
        SobieskiStructure(),
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
    ]
    design_space = SobieskiProblem().design_space
    idf = IDF(disciplines, "y_4", design_space, start_at_equilibrium=True)
    coupling_names = [
        "y_12",
        "y_14",
        "y_21",
        "y_23",
        "y_24",
        "y_31",
        "y_32",
        "y_34",
    ]
    current_couplings = idf.design_space.get_current_value(as_dict=True)
    ref_couplings = SobieskiProblem().get_default_inputs_equilibrium()
    for coupling_name in coupling_names:
        residual = np.linalg.norm(
            current_couplings[coupling_name] - ref_couplings[coupling_name]
        ) / np.linalg.norm(ref_couplings[coupling_name])
        assert residual < 1e-3


def test_grammar_type():
    """Check that the grammar type is correctly used."""
    discipline = AnalyticDiscipline({"y": "x"})
    design_space = DesignSpace()
    design_space.add_variable("x")
    grammar_type = discipline.SIMPLE_GRAMMAR_TYPE
    formulation = IDF(
        [discipline], "y", design_space, grammar_type=grammar_type, n_processes=2
    )
    assert formulation._parallel_exec.grammar_type == grammar_type
