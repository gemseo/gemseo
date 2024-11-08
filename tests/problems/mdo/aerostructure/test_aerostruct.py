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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import unittest
from math import exp

import numpy as np

from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.problems.mdo.aerostructure.aerostructure import Aerodynamics
from gemseo.problems.mdo.aerostructure.aerostructure import Mission
from gemseo.problems.mdo.aerostructure.aerostructure import Structure
from gemseo.problems.mdo.aerostructure.aerostructure import get_inputs
from gemseo.problems.mdo.aerostructure.aerostructure_design_space import (
    AerostructureDesignSpace,
)
from gemseo.scenarios.mdo_scenario import MDOScenario


class TestAerostructure(unittest.TestCase):
    """Test linearization of Aerostructure class."""

    @staticmethod
    def get_xzy():
        """Generate initial solution."""
        drag = np.zeros(1)
        forces = np.zeros(1)
        lift = np.zeros(1)
        mass = np.zeros(1)
        displ = np.zeros(1)
        reserve_fact = np.zeros(1)
        sweep = np.zeros(1)
        thick_airfoils = np.zeros(1)
        thick_panels = np.zeros(1)
        return (
            drag,
            forces,
            lift,
            mass,
            displ,
            reserve_fact,
            sweep,
            thick_airfoils,
            thick_panels,
        )

    @staticmethod
    def get_input_data_linearization():
        """Generate a point at which the problem is linearized."""
        return {
            "drag": np.array([1.0]),
            "forces": np.array([1.0]),
            "lift": np.array([1.0]),
            "mass": np.array([1.0]),
            "displ": np.array([1.0]),
            "reserve_fact": np.array([1.0]),
            "sweep": np.array([1.0]),
            "thick_airfoils": np.array([1.0]),
            "thick_panels": np.array([1.0]),
        }

    def test_run_aero(self) -> None:
        """Evaluate discipline Aero."""
        aero = Aerodynamics()
        aero.execute()
        self.assertAlmostEqual(aero.io.data["drag"][0], 19.60000077, 8)
        self.assertAlmostEqual(aero.io.data["forces"][0], 10.0, 8)
        self.assertAlmostEqual(aero.io.data["lift"][0], -0.00026667, 8)

    def test_run_struct(self) -> None:
        """Evaluate discipline Struct."""
        struct = Structure()
        struct.execute()
        self.assertAlmostEqual(struct.io.data["mass"][0], 200300.00008573389, 10)
        self.assertAlmostEqual(struct.io.data["reserve_fact"][0], 46.1, 10)
        self.assertAlmostEqual(struct.io.data["displ"][0], 3.0, 10)

    def test_run_mission(self) -> None:
        """Evaluate objective function."""
        mission = Mission()
        design_space = AerostructureDesignSpace()
        indata = design_space.get_current_value(as_dict=True)
        indata["lift"] = np.array([exp(-11)])
        indata["mass"] = np.array([exp(1)])
        indata["drag"] = np.array([1])
        mission.execute(indata)
        obj = mission.io.data["range"]
        self.assertAlmostEqual(obj, 4915369.8826625682, 10)

    def test_jac_mission(self) -> None:
        """Test linearization of objective and constraints."""
        mission = Mission()
        indata = TestAerostructure.get_input_data_linearization()
        assert mission.check_jacobian(indata, derr_approx="complex_step", step=1e-30)

    def test_jac_aerodynamics(self) -> None:
        """Test linearization of discipline Aerodynamics."""
        aero = Aerodynamics()
        indata = TestAerostructure.get_input_data_linearization()
        assert aero.check_jacobian(indata, derr_approx="complex_step", step=1e-30)

    def test_jac_structure(self) -> None:
        """Test linearization of discipline Structure."""
        struct = Structure()
        indata = TestAerostructure.get_input_data_linearization()
        assert struct.check_jacobian(indata, derr_approx="complex_step", step=1e-30)
        assert struct.check_jacobian(indata, derr_approx="complex_step", step=1e-30)
        assert struct.check_jacobian(indata, derr_approx="complex_step", step=1e-30)

    def test_mda_gauss_seidel_jac(self) -> None:
        """Test linearization of GS MDA."""
        aerodynamics = Aerodynamics()
        structure = Structure()
        mission = Mission()
        indata = TestAerostructure.get_input_data_linearization()
        disciplines = [aerodynamics, structure, mission]
        mda = MDAGaussSeidel(disciplines)
        mda.execute(indata)
        mda.settings.tolerance = 1e-14
        mda.max_iter = 40
        indata = mda.io.data
        for discipline in disciplines:
            assert discipline.check_jacobian(
                indata, derr_approx="complex_step", step=1e-30
            )
        assert mda.check_jacobian(
            indata, threshold=1e-4, derr_approx="complex_step", step=1e-30
        )

    def test_mda_jacobi_jac(self) -> None:
        """Test linearization of Jacobi MDA."""
        aerodynamics = Aerodynamics()
        structure = Structure()
        mission = Mission()
        indata = TestAerostructure.get_input_data_linearization()
        disciplines = [aerodynamics, structure, mission]
        mda = MDAJacobi(disciplines)
        mda.settings.tolerance = 1e-14
        mda.max_iter = 40
        assert mda.check_jacobian(indata, derr_approx="complex_step", step=1e-30)

    def test_residual_form_jacs(self) -> None:
        """"""
        aerodynamics = Aerodynamics()
        structure = Structure()
        mission = Mission()
        disciplines = [aerodynamics, structure, mission]
        indata = TestAerostructure.get_input_data_linearization()
        for disc in disciplines:
            assert disc.check_jacobian(indata, derr_approx="complex_step", step=1e-30)

    def test_get_inputs(self) -> None:
        get_inputs()
        get_inputs("drag", "forces")
        get_inputs("bad_inputs")


class TestAerostructureScenarios(unittest.TestCase):
    """Test optimization scenarios."""

    # Reference results
    x_ref = np.array((25.0, 20.0, 27.24051915830433))

    @staticmethod
    def create_functional_disciplines():
        """"""
        return [Aerodynamics(), Structure(), Mission()]

    @staticmethod
    def build_scenario(disciplines, formulation_name="MDF"):
        """Build a scenario in functional form with a given formulation.

        :param disciplines: list of disciplines
        :param formulation: name of the formulation (Default value = 'MDF')
        """
        design_space = AerostructureDesignSpace()
        return MDOScenario(
            disciplines,
            "range",
            design_space,
            formulation_name=formulation_name,
        )

    @staticmethod
    def build_and_run_scenario(formulation, algo, lin_method="complex_step"):
        """Create a scenario with given formulation, solver and linearization method,
        and solve it.

        :param formulation: param algo:
        :param lin_method: Default value = 'complex_step')
        :param algo:
        """
        disciplines = TestAerostructureScenarios.create_functional_disciplines()
        scenario = TestAerostructureScenarios.build_scenario(disciplines, formulation)
        scenario.set_differentiation_method(lin_method)

        # add constraints
        scenario.add_constraint("c_lift")
        scenario.add_constraint("c_rf", constraint_type="ineq")
        # run the optimizer
        scenario.execute(algo_name=algo, max_iter=10)
        obj_opt = scenario.optimization_result.f_opt
        xopt = scenario.design_space.get_current_value(as_dict=True)
        sweep = xopt["sweep"]
        thick_airfoils = xopt["thick_airfoils"]
        thick_panels = xopt["thick_panels"]
        x_opt = np.concatenate((thick_airfoils, thick_panels, sweep))

        if formulation == "MDF":
            scenario.formulation.mda.plot_residual_history(save=False)

        return obj_opt, x_opt

    def test_exec_mdf_slsqp_usergrad(self) -> None:
        """Scenario with MDF formulation, solver SLSQP and analytical gradients."""
        f_ref = 3733.1194596682094
        obj_opt, x_opt = TestAerostructureScenarios.build_and_run_scenario(
            "MDF", "SLSQP", lin_method="user"
        )
        rel_err = np.linalg.norm(x_opt - self.x_ref) / np.linalg.norm(self.x_ref)
        self.assertAlmostEqual(obj_opt, f_ref, 1)
        self.assertAlmostEqual(rel_err, 0.0, 4)

    def test_exec_mdf_slsqp_cplx_grad(self) -> None:
        """Scenario with MDF formulation, solver SLSQP and complex step."""
        f_ref = 3733.1202599332164
        obj_opt, x_opt = TestAerostructureScenarios.build_and_run_scenario(
            "MDF", "SLSQP"
        )
        rel_err = np.linalg.norm(x_opt - self.x_ref) / np.linalg.norm(self.x_ref)
        self.assertAlmostEqual(obj_opt, f_ref, 1)
        self.assertAlmostEqual(rel_err, 0.0, 4)

    def test_exec_idf_slsqp_cplxstep(self) -> None:
        """Scenario with IDF formulation, solver SLSQP and complex step."""
        f_ref = 3733.1202599332164
        obj_opt, x_opt = TestAerostructureScenarios.build_and_run_scenario(
            "IDF", "SLSQP"
        )
        # vector of design variables contains y targets at the end
        x_opt = x_opt[:3]
        rel_err = np.linalg.norm(x_opt - self.x_ref) / np.linalg.norm(self.x_ref)
        self.assertAlmostEqual(obj_opt, f_ref, 3)
        self.assertAlmostEqual(rel_err, 0.0, 4)

    def test_exec_idf_slsqp_usergrad(self) -> None:
        """Scenario with IDF formulation, solver SLSQP and analytical gradients."""
        f_ref = 3733.1201817980523
        obj_opt, x_opt = TestAerostructureScenarios.build_and_run_scenario(
            "IDF", "SLSQP", lin_method="user"
        )
        # vector of design variables contains y targets at the end
        x_opt = x_opt[:3]
        rel_err = np.linalg.norm(x_opt - self.x_ref) / np.linalg.norm(self.x_ref)
        self.assertAlmostEqual(obj_opt, f_ref, 1)
        self.assertAlmostEqual(rel_err, 0.0, 4)
