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

import re

import numpy as np
import pytest

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.core.functions.consistency_constraint import ConsistencyConstraint
from gemseo.formulations.idf import IDF
from gemseo.formulations.idf_settings import IDF_Settings
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.evaluation import EvaluationScenario


def test_build_func_from_disc() -> None:
    """"""
    pb = SobieskiProblem("complex128")
    disciplines = [
        SobieskiMission("complex128"),
        SobieskiAerodynamics("complex128"),
        SobieskiPropulsion("complex128"),
        SobieskiStructure("complex128"),
    ]
    problem = OptimizationProblem(pb.design_space)
    idf = IDF(problem, disciplines)
    problem.objective = idf.create_objective(["y_4"])
    assert idf.all_couplings == idf.coupling_structure.all_couplings

    x_names = idf.problem.design_space.variable_names
    x_dict = pb.get_default_inputs(x_names)
    x_vect = np.concatenate([x_dict[k] for k in x_names])

    for c_name in ["g_1", "g_2", "g_3"]:
        constraint = idf.create_constraint(
            [c_name], constraint_type=ArrayFunction.ConstraintType.INEQ
        )
        problem.add_constraint(constraint)
    opt = idf.problem
    opt.objective.check_grad(x_vect, "ComplexStep", 1e-30, error_max=1e-4)
    for cst in opt.constraints:
        cst.check_grad(x_vect, "ComplexStep", 1e-30, error_max=1e-4)

    for func_name in list(pb.get_default_inputs().keys()):
        if func_name.startswith("Y"):
            func = idf._build_func_from_outputs([func_name])
            func.check_grad(x_vect, "ComplexStep", 1e-30, error_max=1e-4)

    for coupl in idf.coupling_structure.strong_couplings:
        func = ConsistencyConstraint([coupl], idf)
        func.check_grad(x_vect, "ComplexStep", 1e-30, error_max=1e-4)


@pytest.mark.parametrize(
    ("options", "expected_feasible"),
    [
        # Without weak coupling targets
        (
            {
                "linearize": False,
                "dtype": "complex128",
                "normalize_cstr": True,
                "eq_tolerance": 1e-4,
                "ineq_tolerance": 1e-4,
                "max_iter": 50,
                "include_weak_coupling_targets": False,
            },
            True,
        ),
        (
            {
                "linearize": True,
                "dtype": "float64",
                "normalize_cstr": True,
                "eq_tolerance": 1e-3,
                "ineq_tolerance": 1e-3,
                "max_iter": 50,
                "include_weak_coupling_targets": False,
            },
            True,
        ),
        # With weak coupling targets
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
            True,
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
) -> None:
    """Test the IDF formulation with an MDOScenario.

    Args:
        options: The options for the generate_idf_scenario fixture.
        expected_feasible: Whether the optimization result is expected to be feasible.
        generate_idf_scenario: Fixture that returns an MDOScenario with an IDF
            formulation with custom arguments.
        caplog: Fixture to access and control log capturing.
    """
    obj_opt, is_feasible = generate_idf_scenario(
        "SLSQP",
        **options,
    )

    if options["max_iter"] == 50:
        # IDF Scenario objective function is normalized by 0.001
        assert 3962.0 < obj_opt * 1000.0 < 3965.0

    assert is_feasible == expected_feasible

    include_weak_coupling_targets = options.get("include_weak_coupling_targets", True)
    if include_weak_coupling_targets and (n_processes := options["n_processes"]) > 1:
        assert (
            f"IDF formulation: running in parallel on {n_processes} processes."
            in caplog.text
        )


def test_fail_idf_no_coupl(generate_idf_scenario) -> None:
    """Test an exception when the coupling variables are not in the Design Space.

    Args:
        generate_idf_scenario: Fixture that returns an MDOScenario with an IDF
            formulation with custom arguments.
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            "IDF formulation: "
            "the variables "
            "'y_12', 'y_14', 'y_21', 'y_23', 'y_24', 'y_31', 'y_32' and 'y_34' "
            "must be added to the design space."
        ),
    ):
        generate_idf_scenario(
            "SLSQP",
            linearize=False,
            dtype="float64",
            normalize_cstr=True,
            remove_coupl_from_ds=True,
        )


def test_idf_start_equilibrium() -> None:
    """Initial value of coupling variables set at equilibrium."""
    disciplines = [
        SobieskiStructure(),
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
    ]
    design_space = SobieskiDesignSpace()
    problem = OptimizationProblem(design_space)
    idf = IDF(problem, disciplines, IDF_Settings(start_at_equilibrium=True))
    problem.objective = idf.create_objective(["y_4"])
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


def test_idf_evaluation_problem() -> None:
    """Check that the consistency constraints are observables
    when IDF uses an EvaluationProblem rather than an OptimizationProblem."""
    disciplines = [
        SobieskiStructure(),
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
    ]
    design_space = SobieskiDesignSpace()
    scenario = EvaluationScenario(
        disciplines, design_space, formulation_settings=IDF_Settings()
    )
    names = [
        "consistency_y_12_y_14",
        "consistency_y_31_y_32_y_34",
        "consistency_y_21_y_23_y_24",
    ]
    assert scenario.formulation.problem.observables.get_names() == names
    assert scenario.formulation.problem.function_names == names
