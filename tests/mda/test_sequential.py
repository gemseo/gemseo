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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path

import numpy as np

from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.mda.sequential_mda import MDAGSNewton
from gemseo.mda.sequential_mda import MDASequential
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.utils import get_y_opt


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

    assert mda.local_data[mda.RESIDUALS_NORM][0] < 1e-6


def test_log_convergence() -> None:
    """Check that the boolean log_convergence is correctly set."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDAGSNewton(disciplines)
    assert not mda.log_convergence
    for sub_mda in mda.mda_sequence:
        assert not sub_mda.log_convergence

    mda = MDAGSNewton(disciplines, log_convergence=True)
    assert mda.log_convergence
    for sub_mda in mda.mda_sequence:
        assert sub_mda.log_convergence

    mda = MDAGSNewton(disciplines)
    mda.log_convergence = True
    assert mda.log_convergence
    for sub_mda in mda.mda_sequence:
        assert sub_mda.log_convergence

    mda.log_convergence = False
    assert not mda.log_convergence
    for sub_mda in mda.mda_sequence:
        assert not sub_mda.log_convergence


def test_parallel_doe(generate_parallel_doe_data) -> None:
    """Test the execution of GaussSeidel in parallel.

    Args:
        generate_parallel_doe_data: Fixture that returns the optimum solution to
            a parallel DOE scenario for a particular `main_mda_name`.
    """
    obj = generate_parallel_doe_data(inner_mda_name="MDAGSNewton")
    assert np.isclose(np.array([-obj]), np.array([608.175]), atol=1e-3)
