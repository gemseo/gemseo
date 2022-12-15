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

import os
import unittest

import numpy as np
import pytest
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.problems.sellar.sellar import C_1
from gemseo.problems.sellar.sellar import C_2
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sellar.sellar import X_LOCAL
from gemseo.problems.sellar.sellar import X_SHARED
from gemseo.problems.sellar.sellar import Y_1
from gemseo.problems.sellar.sellar import Y_2
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace


@pytest.mark.usefixtures("tmp_wd")
class TestSellar(unittest.TestCase):
    """Test linearization of Sellar class."""

    @staticmethod
    def get_xzy():
        """Generate initial solution."""
        x_local = np.array([0.0])
        x_shared = np.array([1.0, 0.0])
        y_0 = np.zeros(1)
        y_1 = np.zeros(1)
        return x_local, x_shared, y_0, y_1

    @staticmethod
    def get_input_data_linearization():
        """Generate a point at which the problem is linearized."""
        return {
            X_LOCAL: np.array([2.1]),
            X_SHARED: np.array([1.2, 3.4]),
            Y_1: np.array([2.135]),
            Y_2: np.array([3.584]),
        }

    def test_run_1(self):
        """Evaluate discipline 1."""
        discipline1 = Sellar1()
        discipline1.execute()
        y_1 = discipline1.get_outputs_by_name(Y_1)
        self.assertAlmostEqual(y_1[0], 0.89442719, 8)

    def test_serialize(self):
        """"""
        fname = "Sellar1.pkl"
        for disc in [Sellar1(), Sellar2(), SellarSystem()]:
            disc.serialize(fname)
            assert os.path.exists(fname)

    def test_jac_sellar_system(self):
        """Test linearization of objective and constraints."""
        system = SellarSystem()
        indata = TestSellar.get_input_data_linearization()
        indata[Y_1] = np.ones([1])
        indata[Y_2] = np.ones([1])
        assert system.check_jacobian(indata, derr_approx="complex_step", step=1e-30)

    def test_jac_sellar1(self):
        """Test linearization of discipline 1."""
        discipline1 = Sellar1()
        indata = TestSellar.get_input_data_linearization()
        assert discipline1.check_jacobian(
            indata, derr_approx="complex_step", step=1e-30
        )

    def test_jac_sellar2(self):
        """Test linearization of discipline 2."""
        discipline2 = Sellar2()
        indata = TestSellar.get_input_data_linearization()
        assert discipline2.check_jacobian(
            indata, derr_approx="complex_step", step=1e-30
        )

        indata[Y_1] = -np.ones([1])
        assert discipline2.check_jacobian(
            indata, derr_approx="complex_step", step=1e-30
        )

        indata[Y_1] = np.zeros([1])
        assert discipline2.check_jacobian(
            indata, derr_approx="complex_step", step=1e-30
        )

    def test_mda_gauss_seidel_jac(self):
        """Test linearization of GS MDA."""
        discipline1 = Sellar1()
        discipline2 = Sellar2()
        system = SellarSystem()
        indata = TestSellar.get_input_data_linearization()
        indata[Y_1] = np.ones([1])
        indata[Y_2] = np.ones([1])

        disciplines = [discipline1, discipline2, system]
        mda = MDAGaussSeidel(
            disciplines, max_mda_iter=100, tolerance=1e-14, over_relax_factor=0.99
        )
        indata = mda.execute(indata)
        for discipline in disciplines:
            assert discipline.check_jacobian(
                indata, derr_approx="complex_step", step=1e-30
            )

        assert mda.check_jacobian(
            indata,
            threshold=1e-4,
            derr_approx="complex_step",
            step=1e-30,
        )

    def test_mda_jacobi_jac(self):
        """Test linearization of Jacobi MDA."""
        discipline1 = Sellar1()
        discipline2 = Sellar2()
        system = SellarSystem()
        indata = self.get_input_data_linearization()
        indata[Y_1] = np.ones([1])
        indata[Y_2] = np.ones([1])

        disciplines = [discipline1, discipline2, system]
        mda = MDAJacobi(disciplines)
        mda.tolerance = 1e-14
        mda.max_iter = 40

        assert mda.check_jacobian(indata, derr_approx="complex_step", step=1e-30)


class TestSellarScenarios(unittest.TestCase):
    """Test optimization scenarios."""

    # Reference results
    x_local_ref = np.array(0.0)
    x_shared_ref = np.array((1.9776, 0.0))
    x_ref = np.hstack((x_local_ref, x_shared_ref))
    y_ref = np.array((1.77763895, 3.75527641))
    f_ref = 3.18339

    @staticmethod
    def create_functional_disciplines():
        """"""
        disciplines = [
            Sellar1(),
            Sellar2(),
            SellarSystem(),
        ]
        return disciplines

    @staticmethod
    def build_scenario(disciplines, formulation="MDF"):
        """Build a scenario in functional form with a given formulation.

        :param disciplines: list of disciplines
        :param formulation: name of the formulation (Default value = 'MDF')
        """
        design_space = SellarDesignSpace()
        scenario = MDOScenario(
            disciplines,
            formulation=formulation,
            objective_name="obj",
            design_space=design_space,
        )
        return scenario

    @staticmethod
    def build_and_run_scenario(formulation, algo, lin_method="complex_step"):
        """Create a scenario with given formulation, solver and linearization method, and
        solve it.

        :param formulation: param algo:
        :param lin_method: Default value = 'complex_step')
        :param algo:
        """
        disciplines = TestSellarScenarios.create_functional_disciplines()
        scenario = TestSellarScenarios.build_scenario(disciplines, formulation)
        scenario.set_differentiation_method(lin_method)

        run_inputs = {"max_iter": 10, "algo": algo}

        # add constraints
        scenario.add_constraint(C_1, "ineq")
        scenario.add_constraint(C_2, "ineq")
        # run the optimizer
        scenario.execute(run_inputs)

        obj_opt = scenario.optimization_result.f_opt
        x_opt = scenario.design_space.get_current_value(as_dict=True)

        x_local = x_opt[X_LOCAL]
        x_shared = x_opt[X_SHARED]
        x_opt = np.concatenate((x_local, x_shared))

        # scenario.post_process("OptHistoryView", show=False, save=True,
        #                      file_path=scenario.formulation.name)
        if formulation == "MDF":
            scenario.formulation.mda.plot_residual_history(save=False)

        return obj_opt, x_opt

    def test_exec_mdf_slsqp_usergrad(self):
        """Scenario with MDF formulation, solver SLSQP and analytical gradients."""
        obj_opt, x_opt = TestSellarScenarios.build_and_run_scenario(
            "MDF", "SLSQP", lin_method="user"
        )
        rel_err = np.linalg.norm(x_opt - self.x_ref) / np.linalg.norm(self.x_ref)
        self.assertAlmostEqual(obj_opt, self.f_ref, 3)
        self.assertAlmostEqual(rel_err, 0.0, 4)

    def test_exec_mdf_slsqp_cplx_grad(self):
        """Scenario with MDF formulation, solver SLSQP and complex step."""
        obj_opt, x_opt = TestSellarScenarios.build_and_run_scenario("MDF", "SLSQP")
        rel_err = np.linalg.norm(x_opt - self.x_ref) / np.linalg.norm(self.x_ref)
        self.assertAlmostEqual(obj_opt, self.f_ref, 3)
        self.assertAlmostEqual(rel_err, 0.0, 4)

    def test_exec_idf_slsqp_cplxstep(self):
        """Scenario with IDF formulation, solver SLSQP and complex step."""
        obj_opt, x_opt = TestSellarScenarios.build_and_run_scenario("IDF", "SLSQP")
        # vector of design variables contains y targets at the end
        x_opt = x_opt[:3]

        rel_err = np.linalg.norm(x_opt - self.x_ref) / np.linalg.norm(self.x_ref)
        self.assertAlmostEqual(obj_opt, self.f_ref, 4)
        self.assertAlmostEqual(rel_err, 0.0, 4)

    def test_exec_idf_slsqp_usergrad(self):
        """Scenario with IDF formulation, solver SLSQP and analytical gradients."""
        obj_opt, x_opt = TestSellarScenarios.build_and_run_scenario(
            "IDF", "SLSQP", lin_method="user"
        )
        # vector of design variables contains y targets at the end
        x_opt = x_opt[:3]

        rel_err = np.linalg.norm(x_opt - self.x_ref) / np.linalg.norm(self.x_ref)
        self.assertAlmostEqual(obj_opt, self.f_ref, 4)
        self.assertAlmostEqual(rel_err, 0.0, 4)
