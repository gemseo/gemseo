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

from functools import partial

import pytest
from numpy.testing import assert_allclose

from gemseo.formulations.mdf import MDF
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.xdsmizer import XDSMizer

from .formulations_basetest import FormulationsBaseTest


class TestMDFFormulation(FormulationsBaseTest):
    """"""

    # Complex step mdf already tested on propane, lighter
    def build_and_run_mdf_scenario_with_constraints(
        self,
        formulation,
        algo="SLSQP",
        linearize=False,
        dtype="complex128",
        **options,
    ):
        """

        :param formulation: param algo:
        :param linearize: Default value = False)
        :param dtype: Default value = "complex128")
        :param algo:

        """
        if not linearize:
            dtype = "complex128"
        scenario = self.build_mdo_scenario(
            formulation, dtype, main_mda_settings=options
        )
        if linearize:
            scenario.set_differentiation_method("user")
        else:
            scenario.set_differentiation_method("complex_step", 1e-30)
        # Set the design constraints
        scenario.add_constraint(["g_1", "g_2", "g_3"], constraint_type="ineq")
        xdsmjson = XDSMizer(scenario).xdsmize()
        assert len(xdsmjson) > 0
        scenario.execute(
            algo_name=algo,
            max_iter=100,
            ftol_rel=1e-10,
            ineq_tolerance=1e-3,
        )
        scenario.print_execution_metrics()
        return scenario.optimization_result.f_opt

    # Tests with SCIPY ONLY ! Other libraries are optional...

    def test_exec_mdf_cstr(self) -> None:
        """"""
        options = {
            "tolerance": 1e-12,
            "max_mda_iter": 50,
            "max_consecutive_unsuccessful_iterations": 50,
            "warm_start": True,
            "use_lu_fact": False,
            "linear_solver_tolerance": 1e-14,
        }

        obj = self.build_and_run_mdf_scenario_with_constraints(
            "MDF", "SLSQP", linearize=True, dtype="float64", **options
        )

        assert_allclose(-obj, 3964.0, atol=1.0, rtol=0)

    def test_getsuboptions(self) -> None:
        self.assertRaises(ValueError, MDF.get_sub_options_grammar)
        self.assertRaises(ValueError, MDF.get_default_sub_option_values)


def test_reset():
    """Check that the optimization problem can be reset.

    See https://gitlab.com/gemseo/dev/gemseo/-/issues/1179.
    """
    design_space = SellarDesignSpace()

    scenario = MDOScenario(
        [Sellar1(), Sellar2(), SellarSystem()],
        "obj",
        design_space,
        formulation_name="MDF",
    )
    initial_current_value = design_space.get_current_value()
    scenario.add_constraint("c_1", constraint_type="ineq")
    scenario.add_constraint("c_2", constraint_type="ineq")
    scenario.execute(algo_name="SLSQP", max_iter=5)
    final_current_value = design_space.get_current_value()

    scenario.formulation.optimization_problem.reset(design_space=True)
    assert_allclose(design_space.get_current_value(), initial_current_value)

    scenario.execute(algo_name="SLSQP", max_iter=5)
    assert_allclose(design_space.get_current_value(), final_current_value)


create_sellar_mdf = partial(
    MDF,
    disciplines=[Sellar1(), Sellar2(), SellarSystem()],
    objective_name="obj",
    design_space=SellarDesignSpace(),
)


def test_mda_settings():
    """Test that the MDA settings are properly handled."""
    mdf = create_sellar_mdf(
        main_mda_name="MDAGaussSeidel",
        main_mda_settings={"max_mda_iter": 13},
    )

    assert isinstance(mdf.mda, MDAGaussSeidel)
    assert mdf.mda.settings.max_mda_iter == 13
    mdf = create_sellar_mdf(
        main_mda_name="MDAGaussSeidel",
        main_mda_settings=MDAGaussSeidel_Settings(max_mda_iter=13),
    )

    assert isinstance(mdf.mda, MDAGaussSeidel)
    assert mdf.mda.settings.max_mda_iter == 13

    msg = (
        "The MDANewtonRaphson settings model has the wrong type: "
        "expected MDANewtonRaphson_Settings, got MDAGaussSeidel_Settings."
    )
    with pytest.raises(TypeError, match=msg):
        mdf = create_sellar_mdf(
            main_mda_name="MDANewtonRaphson",
            main_mda_settings=MDAGaussSeidel_Settings(max_mda_iter=13),
        )
