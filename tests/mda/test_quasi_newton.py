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
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from numpy import array
from numpy import linalg

from gemseo.mda.quasi_newton import MDAQuasiNewton
from gemseo.problems.sellar.sellar import X_SHARED
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sellar.sellar import get_y_opt

from .test_gauss_seidel import SelfCoupledDisc

SELLAR_Y_REF = array([0.80004953, 1.79981434])


def test_quasi_newton_invalid_method():
    """Test invalid method names for quasi Newton."""
    with pytest.raises(
        ValueError, match="Method 'unknown_method' is not a valid quasi-Newton method."
    ):
        MDAQuasiNewton([Sellar1(), Sellar2()], method="unknown_method")


def test_broyden_sellar():
    """Test the execution of quasi-Newton on Sellar."""
    mda = MDAQuasiNewton([Sellar1(), Sellar2()], method=MDAQuasiNewton.BROYDEN1)
    mda.reset_history_each_run = True
    mda.execute()
    assert mda.residual_history[-1] < 1e-5
    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-3

    mda.warm_start = True
    mda.execute({X_SHARED: mda.default_inputs[X_SHARED] + 0.1})


def test_hybrid_sellar():
    """Test the execution of quasi-Newton on Sellar."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDAQuasiNewton(disciplines, use_gradient=True)

    mda.execute()

    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-4


def test_lm_sellar():
    """Test the execution of quasi-Newton on Sellar."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDAQuasiNewton(
        disciplines, method=MDAQuasiNewton.LEVENBERG_MARQUARDT, use_gradient=True
    )
    mda.execute()

    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-4


def test_dfsane_sellar():
    """Test the execution of quasi-Newton on Sellar."""
    mda = MDAQuasiNewton([Sellar1(), Sellar2()], method=MDAQuasiNewton.DF_SANE)
    mda.execute()

    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-3
    with pytest.raises(
        ValueError, match="Method 'unknown_method' is not a valid quasi-Newton method."
    ):
        MDAQuasiNewton([Sellar1(), Sellar2()], method="unknown_method")


def test_broyden_sellar2():
    """Test the execution of quasi-Newton on Sellar."""
    disciplines = [Sellar1(), SellarSystem()]
    mda = MDAQuasiNewton(disciplines, method=MDAQuasiNewton.BROYDEN1)
    mda.reset_history_each_run = True
    mda.execute()

    assert mda.local_data[mda.RESIDUALS_NORM][0] < 1e-6


def test_self_coupled():
    """Test MDAQuasiNewton with a self-coupled discipline."""
    sc_disc = SelfCoupledDisc()
    mda = MDAQuasiNewton([sc_disc], tolerance=1e-14, max_mda_iter=40)
    out = mda.execute()
    assert abs(out["y"] - 2.0 / 3.0) < 1e-6
