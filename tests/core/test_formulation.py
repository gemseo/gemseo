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
import unittest

import numpy as np
import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.core.chain import MDOChain
from gemseo.core.discipline import MDODiscipline
from gemseo.core.formulation import MDOFormulation
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from numpy.dual import norm


class TestMDOFormulation(unittest.TestCase):
    """"""

    def test_get_generator(self):
        """"""
        sm = SobieskiMission()
        ds = SobieskiProblem().design_space
        f = MDOFormulation([sm], "y_4", ds)
        args = ["toto"]
        self.assertRaises(Exception, f._get_generator_with_inputs, *args)

        self.assertRaises(Exception, f._get_generator_from, *args)

    def test_cstrs(self):
        """"""
        sm = SobieskiMission()
        ds = SobieskiProblem().design_space
        f = MDOFormulation([sm], "y_4", ds)
        prob = f.opt_problem
        assert not prob.has_constraints()
        f.add_constraint("y_4", constraint_name="toto")
        assert f.opt_problem.constraints[-1].name == "toto"

    #     def test_disciplines_runinputs(self):
    #         sm = SobieskiMission()
    #         rid = SobieskiProblem().get_default_inputs(sm.get_input_data_names())
    #         f = MDOFormulation('CustomFormulation', [sm], rid, "y_4", ["x_shared"])
    #         inpt = f.get_discipline_run_inputs(sm)
    #         for k in sm.get_input_data_names():
    #             assert(k in inpt)
    #         for k in inpt:
    #             assert(k in sm.get_input_data_names())
    #
    #         gt_rid = f.get_reference_input_data()
    #         for k in rid:
    #             assert k in gt_rid
    #
    #         self.assertRaises(Exception, MDOFormulation, [], rid,
    #                           "y_4", ["x_shared"])
    #         self.assertRaises(Exception, MDOFormulation, None, rid,
    #                           "y_4", ["x_shared"])
    #
    #         self.assertRaises(Exception, f.get_discipline_run_inputs, None)

    def test_jac_sign(self):
        """Check the evaluation and linearization of the sinus MDOFunction."""
        # TODO: this test should be removed as it does not check MDOFormulation.
        sm = SobieskiMission()
        design_space = DesignSpace()
        design_space.add_variable("x_shared")
        f = MDOFormulation([sm], "y_4", design_space)

        g = MDOFunction(
            math.sin,
            name="G",
            f_type="ineq",
            jac=math.cos,
            expr="sin(x)",
            args=["x", "y"],
        )
        f.opt_problem.objective = g

        obj = f.opt_problem.objective
        self.assertAlmostEqual(obj(math.pi / 2), 1.0, 9)
        self.assertAlmostEqual(obj.jac(0.0), 1.0, 9)

    def test_get_x0(self):
        """"""
        _ = MDOFormulation([SobieskiMission()], "y_4", SobieskiProblem().design_space)

    def test_add_user_defined_constraint_error(self):
        """Check that an error is raised when adding a constraint with wrong type."""
        sm = SobieskiMission()
        design_space = DesignSpace()
        design_space.add_variable("x_shared")
        f = MDOFormulation([sm], "y_4", design_space)
        self.assertRaises(Exception, f.add_constraint, "y_4", "None", "None")

    # =========================================================================
    #     def test_add_user_defined_constraint(self):
    #         sm = SobieskiMission()
    #         design_space = DesignSpace()
    #         design_space.add_variable("x_shared", 1)
    #
    #         f = MDOFormulation([sm], "y_4", design_space)
    #         _, add_to = f.add_constraint(
    #             'y_4', constraint_type="ineq", constraint_name="InEq")
    #         assert add_to
    # =========================================================================

    def test_get_values_array_from_dict(self):
        """"""
        a = concatenate_dict_of_arrays_to_array({}, [])
        self.assertIsInstance(a, type(np.array([])))

    def test_get_mask_from_datanames(self):
        """"""
        a = MDOFormulation._get_mask_from_datanames(["y_1", "y_2", "y_3"], ["y_2"])[0][
            0
        ]
        self.assertEqual(a, 1)

    def test_x_mask(self):
        """"""
        sm = SobieskiMission()
        rid = SobieskiProblem().get_default_inputs(sm.get_input_data_names())
        dvs = ["x_shared", "y_14"]

        design_space = DesignSpace()
        design_space.add_variable("x_shared", 4)
        design_space.add_variable("y_14", 4)
        f = MDOFormulation([sm], "y_4", design_space)

        x = np.concatenate([rid[n] for n in dvs])
        c = f.mask_x_swap_order(dvs, x, dvs)
        expected = np.array(
            [
                0.05,
                4.5e04,
                1.6,
                5.5,
                55.0,
                1000.0,
                50606.9741711000024,
                7306.20262123999964,
            ]
        )
        assert norm(c - expected) < 1e-14
        x_values_dict = f._get_dv_indices(dvs)
        assert x_values_dict == {"x_shared": (0, 4, 4), "y_14": (4, 8, 4)}

        with pytest.raises(KeyError):
            f.mask_x_swap_order(dvs + ["toto"], x)

        ff = f.mask_x_swap_order(
            ["x_shared"],
            x_vect=np.zeros(19),
            all_data_names=design_space.variables_names,
        )
        assert (ff == np.zeros(4)).all()

        design_space.remove_variable("x_shared")
        design_space.add_variable("x_shared", 10)
        self.assertRaises(IndexError, f.mask_x_swap_order, dvs, x)

    def test_remove_sub_scenario_dv_from_ds(self):
        ds2 = DesignSpace()
        ds2.add_variable("y_14")
        ds2.add_variable("x")
        ds1 = DesignSpace()
        ds1.add_variable("x")
        sm = SobieskiMission()
        s1 = MDOScenario([sm], "IDF", "y_4", ds1)
        f2 = MDOFormulation([sm, s1], "y_4", ds2)
        assert "x" in f2.design_space.variables_names
        f2._remove_sub_scenario_dv_from_ds()
        assert "x" not in f2.design_space.variables_names

    def test_get_obj(self):
        """"""
        sm = SobieskiMission()
        dvs = ["x_shared", "y_14"]

        design_space = DesignSpace()
        for name in dvs:
            design_space.add_variable(name)

        f = MDOFormulation([sm], "Y5", design_space)
        self.assertRaises(Exception, lambda: f.get_objective())

    def test_get_expected_workflow(self):
        """"""
        sm = SobieskiMission()
        ds = SobieskiProblem().design_space
        f = MDOFormulation([sm], "Y5", ds)
        self.assertRaises(Exception, f.get_expected_workflow)


def test_grammar_type():
    """Check that the grammar type is correctly stored."""
    discipline = AnalyticDiscipline({"y": "x"})
    design_space = DesignSpace()
    design_space.add_variable("x")
    formulation = MDOFormulation(
        [discipline], "y", design_space, grammar_type="a_grammar_type"
    )
    assert formulation._grammar_type == "a_grammar_type"


def test_remove_unused_variable_logger(caplog):
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
    formulation = MDOFormulation([y1, y2, y3], "y2", design_space)
    formulation._remove_unused_variables()
    assert (
        "Variable toto was removed from the Design Space, it is not an input of any "
        "discipline." in caplog.text
    )


@pytest.mark.parametrize(
    "recursive, expected", [(False, {"d1", "chain2"}), (True, {"d1", "d2", "d3"})]
)
def test_get_sub_disciplines_recursive(recursive, expected):
    """Test the recursive option of get_sub_disciplines.

    Args:
        recursive: Whether to list sub-disciplines recursively.
        expected: The expected disciplines.
    """
    d1 = MDODiscipline("d1")
    d2 = MDODiscipline("d2")
    d3 = MDODiscipline("d3")
    chain1 = MDOChain([d3], "chain1")
    chain2 = MDOChain([d2, chain1], "chain2")
    chain3 = MDOChain([d1, chain2], "chain3")
    design_space = DesignSpace()

    formulation = MDOFormulation([chain3], "foo", design_space)

    classes = [
        discipline.name
        for discipline in formulation.get_sub_disciplines(recursive=recursive)
    ]

    assert set(classes) == expected
