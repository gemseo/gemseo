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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import unittest

from numpy import array
from numpy import complex128
from numpy import concatenate
from numpy import float64
from numpy import ones
from numpy import zeros
from numpy.linalg import norm

from gemseo.problems.mdo.propane.propane import PropaneComb1
from gemseo.problems.mdo.propane.propane import PropaneComb2
from gemseo.problems.mdo.propane.propane import PropaneComb3
from gemseo.problems.mdo.propane.propane import PropaneReaction
from gemseo.problems.mdo.propane.propane import get_design_space
from gemseo.scenarios.mdo_scenario import MDOScenario


class TestPropaneScenario(unittest.TestCase):
    """"""

    def get_current_x(self):
        """"""
        return get_design_space().get_current_value()

    def get_inputs_by_names(self, data_names):
        """

        :param data_names:

        """
        data_dict = self.get_current_x()
        return [data_dict[k] for k in data_names]

    def get_x0(self, scenario):
        """

        :param scenario:

        """
        data_names = scenario.get_optim_variable_names()
        data = [self.io_data.data[name] for name in data_names]
        return concatenate(data)

    def build_mdo_scenario(self, formulation_name="MDF"):
        """

        :param formulation: Default value = 'MDF')

        """
        disciplines = [
            PropaneComb1(),
            PropaneComb2(),
            PropaneComb3(),
            PropaneReaction(),
        ]
        design_space = get_design_space()
        return MDOScenario(
            disciplines,
            "obj",
            design_space,
            formulation_name=formulation_name,
        )

    def build_and_run_scenario(self, formulation, algo, lin_method="complex_step"):
        """

        :param formulation: param algo:
        :param lin_method: Default value = 'complex_step')
        :param algo:

        """
        scenario = self.build_mdo_scenario(formulation)
        scenario.set_differentiation_method(lin_method)
        # add constraints

        scenario.add_constraint(["f_2", "f_6"], constraint_type="ineq")
        scenario.add_constraint(["f_7", "f_9"], constraint_type="ineq")

        # run the optimizer
        scenario.execute(algo_name=algo, max_iter=50)
        obj_opt = scenario.optimization_result.f_opt

        x_opt = scenario.design_space.get_current_value()
        return obj_opt, x_opt

    def test_init_mdf(self) -> None:
        """"""
        self.build_mdo_scenario()

    def test_init_idf(self) -> None:
        """"""
        self.build_mdo_scenario("IDF")

    def test_exec_mdf_mma(self) -> None:
        """"""
        obj_opt, x_opt = self.build_and_run_scenario("MDF", "SLSQP")
        self.assertAlmostEqual(obj_opt, 0, 2)
        x_ref = array((1.378887, 18.426810, 1.094798, 0.931214))
        rel_err = norm(x_opt - x_ref) / norm(x_ref)
        self.assertAlmostEqual(rel_err, 0, 3)


# =========================================================================
#     def test_exec_idf_SLSQP(self):
#         obj_opt, x_opt = self.build_and_run_scenario('IDF',
#                                                      'SLSQP')
#         logger.debug("obj with constraints=" + str(obj_opt))
#         logger.debug("x_opt=" + str(x_opt))
#         self.assertAlmostEqual(obj_opt, 0., 4)
#         x_opt = x_opt[:4]
#         x_ref = array((1.378887, 18.426810, 1.094798, 0.931214))
#         rel_err = linalg.norm(
#             x_opt - x_ref) / linalg.norm(x_ref)
#
#         self.assertAlmostEqual(rel_err, 0, 4)
# =========================================================================


class TestPropaneCombustion(unittest.TestCase):
    """"""

    def test_init_1(self) -> None:
        """"""
        PropaneComb1()

    def test_init_2(self) -> None:
        """"""
        PropaneComb2()

    def test_init_3(self) -> None:
        """"""
        PropaneComb3()

    def get_xy(self):
        """"""
        y_1 = zeros(2, dtype=complex128)
        y_2 = zeros(2, dtype=complex128)
        y_3 = zeros(3, dtype=complex128)
        x_shared = ones(4, dtype=float64)
        return y_1, y_2, y_3, x_shared

    def get_current_x(self):
        """"""
        y_1, y_2, y_3, x_shared = self.get_xy()
        return {"y_1": y_1, "y_2": y_2, "y_3": y_3, "x_shared": x_shared}

    def test_run_1(self) -> None:
        """"""
        pc = PropaneComb1()
        pc.execute(self.get_current_x())
        y_1 = pc.io.data["y_1"]
        self.assertAlmostEqual(y_1[0], 1.0, 10)
        self.assertAlmostEqual(y_1[1], 2.0, 10)

    def test_run_2(self) -> None:
        """"""
        pc = PropaneComb2()
        pc.execute(self.get_current_x())
        y_2 = pc.io.data["y_2"]
        self.assertAlmostEqual(y_2[0], 2.0, 10)
        self.assertAlmostEqual(y_2[1], 0.058860363180964305, 10)

    def test_run_3(self) -> None:
        """"""
        pc = PropaneComb3()
        pc.execute(self.get_current_x())
        y_3 = pc.io.data["y_3"]
        self.assertAlmostEqual(y_3[0], 38.0, 10)
        self.assertAlmostEqual(y_3[1], 0.029430181590482152, 10)
        self.assertAlmostEqual(y_3[2], 47.088290544771446, 10)

    def test_run_reac(self) -> None:
        """"""
        pc = PropaneReaction()
        indata = self.get_current_x()
        indata["y_1"] = ones([2])
        indata["y_2"] = ones([2])
        indata["y_3"] = ones([3])
        pc.execute(indata)
        f = pc.io.data["obj"]
        self.assertAlmostEqual(f, -16.973665961010276, 10)
