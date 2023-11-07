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

import numpy as np
import pytest

from gemseo.core.mdo_scenario import MDOScenario
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.problems.sellar.sellar import C_1
from gemseo.problems.sellar.sellar import C_2
from gemseo.problems.sellar.sellar import WITH_2D_ARRAY
from gemseo.problems.sellar.sellar import X_LOCAL
from gemseo.problems.sellar.sellar import X_SHARED
from gemseo.problems.sellar.sellar import Y_1
from gemseo.problems.sellar.sellar import Y_2
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace


@pytest.fixture()
def input_data():
    """Generate a point at which the problem is linearized."""
    x_shared = [1.2, 3.4]
    if WITH_2D_ARRAY:
        # This handles running the test suite for checking data conversion.
        x_shared = [x_shared]
    return {
        X_LOCAL: 2.1,
        X_SHARED: np.array(x_shared),
        Y_1: 2.135,
        Y_2: 3.584,
    }


def test_run_1():
    """Evaluate discipline 1."""
    discipline1 = Sellar1()
    discipline1.execute()
    y_1 = discipline1.get_outputs_by_name(Y_1)
    assert y_1 == pytest.approx(0.89442719, 1.0e-8)


def test_serialize(tmp_wd):
    """Verify the serialization."""
    fname = "Sellar1.pkl"
    for disc in [Sellar1(), Sellar2(), SellarSystem()]:
        disc.to_pickle(fname)
        assert os.path.exists(fname)


def test_jac_sellar_system(input_data, sellar_disciplines):
    """Test linearization of objective and constraints."""
    system = sellar_disciplines.sellar_system
    input_data[Y_1] = 1.0
    input_data[Y_2] = 1.0
    assert system.check_jacobian(input_data, derr_approx="complex_step", step=1e-30)


def test_jac_sellar1(input_data, sellar_disciplines):
    """Test linearization of discipline 1."""
    discipline1 = sellar_disciplines.sellar1
    assert discipline1.check_jacobian(
        input_data, derr_approx="complex_step", step=1e-30
    )


def test_jac_sellar2(input_data, sellar_disciplines):
    """Test linearization of discipline 2."""
    discipline2 = sellar_disciplines.sellar2
    assert discipline2.check_jacobian(
        input_data, derr_approx="complex_step", step=1e-30
    )

    input_data[Y_1] = -1.0
    assert discipline2.check_jacobian(
        input_data, derr_approx="complex_step", step=1e-30
    )

    input_data[Y_1] = 0.0
    assert discipline2.check_jacobian(
        input_data, derr_approx="complex_step", step=1e-30
    )


def test_mda_gauss_seidel_jac(input_data, sellar_disciplines):
    """Test linearization of GS MDA."""
    discipline1 = sellar_disciplines.sellar1
    discipline2 = sellar_disciplines.sellar2
    system = SellarSystem()
    input_data[Y_1] = 1.0
    input_data[Y_2] = 1.0

    disciplines = [discipline1, discipline2, system]
    mda = MDAGaussSeidel(
        disciplines, max_mda_iter=100, tolerance=1e-14, over_relax_factor=0.99
    )
    input_data = mda.execute(input_data)
    for discipline in disciplines:
        assert discipline.check_jacobian(
            input_data, derr_approx="complex_step", step=1e-30
        )

    assert mda.check_jacobian(
        input_data,
        threshold=1e-4,
        derr_approx="complex_step",
        step=1e-30,
    )


def test_mda_jacobi_jac(input_data, sellar_disciplines):
    """Test linearization of Jacobi MDA."""
    discipline1 = sellar_disciplines.sellar1
    discipline2 = sellar_disciplines.sellar2
    system = SellarSystem()
    input_data[Y_1] = 1.0
    input_data[Y_2] = 1.0

    disciplines = [discipline1, discipline2, system]
    mda = MDAJacobi(disciplines)
    mda.tolerance = 1e-14
    mda.max_iter = 40

    assert mda.check_jacobian(input_data, derr_approx="complex_step", step=1e-30)


X_LOCAL_REF = np.array(0.0)
X_SHARED_REF = np.array((1.9776, 0.0))
X_REF = np.hstack((X_LOCAL_REF, X_SHARED_REF))
F_REF = 3.18339


@pytest.mark.parametrize(
    ("formulation", "algo", "lin_method"),
    [
        ("MDF", "SLSQP", "user"),
        ("MDF", "SLSQP", "complex_step"),
        ("IDF", "SLSQP", "complex_step"),
        ("IDF", "SLSQP", "user"),
    ],
)
def test_exec(formulation, algo, lin_method, sellar_disciplines):
    """Scenario with MDF formulation, solver SLSQP and analytical gradients."""
    scenario = MDOScenario(
        sellar_disciplines,
        formulation=formulation,
        objective_name="obj",
        design_space=SellarDesignSpace(),
    )
    scenario.set_differentiation_method(lin_method)
    scenario.add_constraint(C_1, "ineq")
    scenario.add_constraint(C_2, "ineq")
    scenario.execute({"max_iter": 10, "algo": algo})

    x_opt = scenario.design_space.get_current_value(as_dict=True)
    x_opt = np.concatenate((x_opt[X_LOCAL], x_opt[X_SHARED]))

    assert scenario.optimization_result.f_opt == pytest.approx(F_REF, rel=0.001)
    assert x_opt == pytest.approx(X_REF, rel=0.0001)
