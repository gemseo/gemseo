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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.linalg import norm

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.chains.chain import MDOChain
from gemseo.core.discipline import Discipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.base_mdo import BaseMDOFormulation
from gemseo.formulations.base_settings import BaseFormulationSettings
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt
from gemseo.formulations.idf_settings import IDF_Settings
from gemseo.formulations.mdf import MDF
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.scenarios.mdo import MDOScenario
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.discipline import get_sub_disciplines
from gemseo.utils.testing.helpers import concretize_classes


class NewMDOFormulationSettings(BaseFormulationSettings): ...


class NewMDOFormulation(BaseMDOFormulation):
    settings_class = NewMDOFormulationSettings

    def get_top_level_disciplines(
        self, include_sub_formulations: bool = False
    ) -> tuple[Discipline, ...]:
        return self.disciplines


@pytest.fixture(scope="module")
def patch_mdo_formulation():
    with concretize_classes(NewMDOFormulation):
        yield


def test_cstrs(patch_mdo_formulation) -> None:
    """"""
    sm = SobieskiMission()
    ds = SobieskiDesignSpace()
    prob = OptimizationProblem(ds)
    f = DisciplinaryOpt(prob, [sm])
    prob.objective = f.create_objective(["y_4"])
    assert not prob.constraints
    constraint = f.create_constraint(["y_4"], constraint_name="toto")
    prob.add_constraint(constraint)
    assert f.problem.constraints[-1].name == "toto"


#     def test_disciplines_runinputs(self):
#         sm = SobieskiMission()
#         rid = SobieskiProblem().get_default_inputs(sm.io.input_grammar)
#         f = BaseMDOFormulation('CustomFormulation', [sm], rid, "y_4", ["x_shared"])
#         inpt = f.get_discipline_run_inputs(sm)
#         for k in sm.io.input_grammar:
#             assert(k in inpt)
#         for k in inpt:
#             assert(k in sm.io.input_grammar)
#
#         gt_rid = f.get_reference_input_data()
#         for k in rid:
#             assert k in gt_rid
#
#         self.assertRaises(Exception, BaseMDOFormulation, [], rid,
#                           "y_4", ["x_shared"])
#         self.assertRaises(Exception, BaseMDOFormulation, None, rid,
#                           "y_4", ["x_shared"])
#
#         self.assertRaises(Exception, f.get_discipline_run_inputs, None)


def test_jac_sign(patch_mdo_formulation) -> None:
    """Check the evaluation and linearization of the sinus MDOFunction."""
    # TODO: this test should be removed as it does not check BaseMDOFormulation.
    sm = SobieskiMission()
    design_space = DesignSpace()
    design_space.add_variable("x_shared")
    problem = OptimizationProblem(design_space)
    f = NewMDOFormulation(problem, [sm])
    problem.objective = f.create_objective(["y_4"])

    g = MDOFunction(
        math.sin,
        name="G",
        f_type=MDOFunction.ConstraintType.INEQ,
        jac=math.cos,
        expr="sin(x)",
        input_names=["x", "y"],
    )
    f.problem.objective = g

    obj = f.problem.objective
    assert obj.evaluate(math.pi / 2) == pytest.approx(1.0, 1.0e-9)
    assert obj.jac(0.0) == pytest.approx(1.0, 1.0e-9)


# =========================================================================
#     def test_add_user_defined_constraint(self):
#         sm = SobieskiMission()
#         design_space = DesignSpace()
#         design_space.add_variable("x_shared", 1)
#
#         f = BaseMDOFormulation([sm], "y_4", design_space)
#         _, add_to = f.add_constraint(
#             'y_4', constraint_type="ineq", constraint_name="InEq")
#         assert add_to
# =========================================================================


def test_get_values_array_from_dict() -> None:
    """"""
    a = concatenate_dict_of_arrays_to_array({}, [])
    assert isinstance(a, type(np.array([])))


def test_x_mask(patch_mdo_formulation) -> None:
    """"""
    sm = SobieskiMission()
    rid = SobieskiProblem().get_default_inputs(sm.io.input_grammar)
    dvs = ["x_shared", "y_14"]

    design_space = DesignSpace()
    design_space.add_variable("x_shared", 4)
    design_space.add_variable("y_14", 4)

    problem = OptimizationProblem(design_space)
    f = NewMDOFormulation(problem, [sm])
    problem.objective = f.create_objective(["y_4"])

    x = np.concatenate([rid[n] for n in dvs])
    c = f.mask_x_swap_order(dvs, x, dvs)
    expected = np.array([
        0.05,
        4.5e04,
        1.6,
        5.5,
        55.0,
        1000.0,
        50606.9741711000024,
        7306.20262123999964,
    ])
    assert norm(c - expected) < 1e-14
    x_values_dict = f._get_dv_indices(dvs)
    assert x_values_dict == {"x_shared": (0, 4, 4), "y_14": (4, 8, 4)}

    with pytest.raises(KeyError):
        f.mask_x_swap_order([*dvs, "toto"], x)

    ff = f.mask_x_swap_order(
        ["x_shared"],
        x_vect=np.zeros(19),
        all_data_names=design_space,
    )
    assert (ff == np.zeros(4)).all()

    design_space.remove_variable("x_shared")
    design_space.add_variable("x_shared", 10)
    with pytest.raises(IndexError):
        f.mask_x_swap_order(dvs, x)


def test_remove_sub_scenario_dv_from_ds() -> None:
    ds2 = DesignSpace()
    ds2.add_variable("y_14")
    ds2.add_variable("x")
    ds1 = DesignSpace()
    ds1.add_variable("x")
    sm = SobieskiMission()
    s1 = MDOScenario([sm], ds1, formulation_settings=IDF_Settings())
    s1.add_objective("y_4")
    problem = OptimizationProblem(ds2)
    f2 = NewMDOFormulation(problem, [sm, s1])
    problem.objective = f2.create_objective(["y_4"])
    assert "x" in f2.design_space
    f2._remove_sub_scenario_dv_from_ds()
    assert "x" not in f2.design_space


def test_remove_unused_variable_logger(patch_mdo_formulation, caplog) -> None:
    """Check that a message is logged when an unused variable is removed.

    Args:
        caplog: Fixture to access and control log capturing.
    """
    y1 = AnalyticDiscipline({"y1": "x1+y2"})
    y2 = AnalyticDiscipline({"y2": "x2+y1"})
    y3 = AnalyticDiscipline({"toto": "x3+y1+y2"})
    design_space = DesignSpace()
    design_space.add_variable("x1")
    design_space.add_variable("x2")
    design_space.add_variable("y1")
    design_space.add_variable("toto")
    problem = OptimizationProblem(design_space)
    formulation = MDF(problem, [y1, y2, y3])
    problem.objective = formulation.create_objective(["y2"])
    formulation._remove_unused_variables()
    assert (
        "Variable toto was removed from the Design Space, it is not an input of "
        "any "
        "discipline." in caplog.text
    )


@pytest.mark.parametrize(
    ("recursive", "expected"), [(False, {"d1", "chain2"}), (True, {"d1", "d2", "d3"})]
)
def test_get_sub_disciplines_recursive(
    patch_mdo_formulation, recursive, expected
) -> None:
    """Test the recursive option of get_sub_disciplines.

    Args:
        recursive: Whether to list sub-disciplines recursively.
        expected: The expected disciplines.
    """
    with concretize_classes(Discipline):
        d1 = Discipline("d1")
        d2 = Discipline("d2")
        d3 = Discipline("d3")
    d1.io.output_grammar.update_from_names(["foo"])
    chain1 = MDOChain([d3], "chain1")
    chain2 = MDOChain([d2, chain1], "chain2")
    chain3 = MDOChain([d1, chain2], "chain3")
    design_space = DesignSpace()
    problem = OptimizationProblem(design_space)
    formulation = NewMDOFormulation(problem, [chain3])
    problem.objective = formulation.create_objective(["foo"])

    classes = [
        discipline.name
        for discipline in get_sub_disciplines(
            formulation.disciplines, recursive=recursive
        )
    ]

    assert set(classes) == expected
