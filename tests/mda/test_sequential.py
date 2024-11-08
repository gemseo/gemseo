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
#
# Copyright 2024 Capgemini
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path

import numpy as np

from gemseo.mda.base_mda import BaseMDA
from gemseo.mda.gs_newton import MDAGSNewton
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.mda.sequential_mda import MDASequential
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.utils import get_y_opt

from .utils import generate_parallel_doe


def test_sequential_mda_sellar(tmp_wd) -> None:
    disciplines = [Sellar1(), Sellar2()]
    mda1 = MDAJacobi(disciplines, max_mda_iter=1)
    mda2 = MDANewtonRaphson(disciplines)
    mda_sequence = [mda1, mda2]

    mda = MDASequential(disciplines, mda_sequence, max_mda_iter=20)
    mda.reset_history_each_run = True
    mda.execute()

    y_ref = np.array([0.80004953, 1.79981434])
    assert np.linalg.norm(y_ref - get_y_opt(mda)) / np.linalg.norm(y_ref) < 1e-4

    mda3 = MDAGSNewton(disciplines, max_mda_iter=4)
    mda3.execute()
    filename = "GS_sellar.pdf"
    mda3.plot_residual_history(filename=filename)

    assert Path(filename).exists
    assert np.linalg.norm(y_ref - get_y_opt(mda3)) / np.linalg.norm(y_ref) < 1e-4

    assert mda.io.data[mda.NORMALIZED_RESIDUAL_NORM][0] < 1e-6


def test_log_convergence() -> None:
    """Check that the boolean log_convergence is correctly set."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDAGSNewton(disciplines)
    assert not mda.settings.log_convergence
    for sub_mda in mda.mda_sequence:
        assert not sub_mda.settings.log_convergence

    mda = MDAGSNewton(disciplines, log_convergence=True)
    assert mda.settings.log_convergence
    for sub_mda in mda.mda_sequence:
        assert sub_mda.settings.log_convergence

    mda = MDAGSNewton(disciplines)
    mda.settings.log_convergence = True
    assert mda.settings.log_convergence
    for sub_mda in mda.mda_sequence:
        assert sub_mda.settings.log_convergence

    mda.settings.log_convergence = False
    assert not mda.settings.log_convergence
    for sub_mda in mda.mda_sequence:
        assert not sub_mda.settings.log_convergence


def test_parallel_doe() -> None:
    """Test the execution of GaussSeidel in parallel."""
    obj = generate_parallel_doe(main_mda_settings={"inner_mda_name": "MDAGSNewton"})
    assert np.isclose(np.array([-obj]), np.array([608.175]), atol=1e-3)


def test_sequential_mda_scaling_method() -> None:
    """Test changing the `scaling` attribute of a sequential MDA.

    Check that the change is propagated to the sub-MDAs.
    """
    disciplines = [Sellar1(), Sellar2()]
    mda1 = MDAJacobi(disciplines, max_mda_iter=1)
    mda2 = MDANewtonRaphson(disciplines)
    mda_sequence = [mda1, mda2]
    mda = MDASequential(disciplines, mda_sequence, max_mda_iter=20)

    mda.scaling = BaseMDA.ResidualScaling.NO_SCALING
    assert mda.scaling == BaseMDA.ResidualScaling.NO_SCALING
    assert mda1.scaling == BaseMDA.ResidualScaling.NO_SCALING
    assert mda2.scaling == BaseMDA.ResidualScaling.NO_SCALING


def test_mda_gs_newton_tolerances() -> None:
    """Check that the `tolerance` arguments are correctly propagated to the sub-MDAs."""
    disciplines = [Sellar1(), Sellar2()]
    tolerance = 1e-10
    linear_solver_tolerance = 1e-15
    mda = MDAGSNewton(
        disciplines,
        tolerance=tolerance,
        linear_solver_tolerance=linear_solver_tolerance,
    )
    assert mda.settings.tolerance == tolerance
    assert mda.settings.linear_solver_tolerance == linear_solver_tolerance
    assert mda.mda_sequence[0].settings.tolerance == tolerance
    assert mda.mda_sequence[1].settings.tolerance == tolerance
    mda1_tol = mda.mda_sequence[1].settings.linear_solver_tolerance
    assert mda1_tol == linear_solver_tolerance
