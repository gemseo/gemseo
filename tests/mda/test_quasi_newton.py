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

import re

import pytest
from numpy import array
from numpy import linalg

from gemseo.mda.quasi_newton import MDAQuasiNewton
from gemseo.mda.quasi_newton import QuasiNewtonMethod
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.problems.mdo.sellar.utils import get_y_opt
from gemseo.problems.mdo.sellar.variables import X_SHARED
from tests.mda import check_iteration_callbacks_clearing
from tests.mda import check_iteration_callbacks_execution

from .test_gauss_seidel import SelfCoupledDisc

SELLAR_Y_REF = array([0.80004953, 1.79981434])


@pytest.mark.parametrize(
    "method",
    [
        QuasiNewtonMethod.BROYDEN1,
        QuasiNewtonMethod.BROYDEN2,
    ],
)
def test_broyden_sellar(method) -> None:
    """Test the execution of quasi-Newton on Sellar with a Broyden method."""
    mda = MDAQuasiNewton([Sellar1(), Sellar2()], method=method)
    mda.reset_history_each_run = True
    mda.execute()
    assert mda.residual_history[-1] < 1e-5
    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-3

    mda.settings.warm_start = True
    mda.execute({X_SHARED: mda.io.input_grammar.defaults[X_SHARED] + 0.1})


def test_hybrid_sellar() -> None:
    """Test the execution of quasi-Newton on Sellar."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDAQuasiNewton(disciplines, use_gradient=True)

    mda.execute()

    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-4


def test_lm_sellar() -> None:
    """Test the execution of quasi-Newton on Sellar."""
    disciplines = [Sellar1(), Sellar2()]
    mda = MDAQuasiNewton(
        disciplines,
        method=QuasiNewtonMethod.LEVENBERG_MARQUARDT,
        use_gradient=True,
    )
    mda.execute()

    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-4


def test_dfsane_sellar() -> None:
    """Test the execution of quasi-Newton on Sellar."""
    mda = MDAQuasiNewton([Sellar1(), Sellar2()], method=QuasiNewtonMethod.DF_SANE)
    mda.execute()

    assert linalg.norm(SELLAR_Y_REF - get_y_opt(mda)) / linalg.norm(SELLAR_Y_REF) < 1e-3


def test_broyden_sellar2() -> None:
    """Test the execution of quasi-Newton on Sellar."""
    disciplines = [Sellar1(), SellarSystem()]
    mda = MDAQuasiNewton(disciplines, method=QuasiNewtonMethod.BROYDEN1)
    mda.reset_history_each_run = True
    mda.execute()

    assert mda.io.data[mda.NORMALIZED_RESIDUAL_NORM][0] < 1e-6


def test_self_coupled() -> None:
    """Test MDAQuasiNewton with a self-coupled discipline."""
    sc_disc = SelfCoupledDisc()
    mda = MDAQuasiNewton([sc_disc], tolerance=1e-14, max_mda_iter=40)
    out = mda.execute()
    assert abs(out["y"] - 2.0 / 3.0) < 1e-6


@pytest.mark.parametrize("method", QuasiNewtonMethod)
def test_methods_supporting_callbacks(method):
    """Test MDAQuasiNewton._METHODS_SUPPORTING_CALLBACKS."""
    mda = MDAQuasiNewton([Sellar1(), SellarSystem()], method=method)
    method_supports_callbacks = method in MDAQuasiNewton._METHODS_SUPPORTING_CALLBACKS
    assert (
        mda.NORMALIZED_RESIDUAL_NORM in mda.io.output_grammar
    ) is method_supports_callbacks


@pytest.mark.parametrize("method", MDAQuasiNewton._METHODS_SUPPORTING_CALLBACKS)
def test_residual_history(sellar_disciplines, method):
    """Test that method supporting callbacks update convergence metrics."""
    mda = MDAQuasiNewton(sellar_disciplines, method=method)
    mda.execute()

    assert len(mda.residual_history) >= 7
    assert len(mda.residual_history) == mda._current_iter
    assert mda.residual_history[-1] <= mda.settings.tolerance


@pytest.mark.parametrize("method", MDAQuasiNewton._METHODS_SUPPORTING_CALLBACKS)
def test_iteration_callbacks_execution(method) -> None:
    """Check the execution of iteration callbacks."""
    check_iteration_callbacks_execution(
        MDAQuasiNewton([Sellar1(), Sellar2()], method=method)
    )


@pytest.mark.parametrize("method", MDAQuasiNewton._METHODS_SUPPORTING_CALLBACKS)
def test_iteration_callbacks_clearing(method) -> None:
    """Check the clearing of iteration callbacks."""
    check_iteration_callbacks_clearing(
        MDAQuasiNewton([Sellar1(), Sellar2()], method=method)
    )


@pytest.mark.parametrize(
    "method",
    sorted(
        set(QuasiNewtonMethod).difference(MDAQuasiNewton._METHODS_SUPPORTING_CALLBACKS)
    ),
)
def test_iteration_callbacks_unsupported(method) -> None:
    """Check the iteration callbacks for unsupported methods."""
    mda = MDAQuasiNewton([Sellar1(), Sellar2()], method=method)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Iteration callbacks are only supported for the methods: "
            f"{MDAQuasiNewton._METHODS_SUPPORTING_CALLBACKS}, not for {method}."
        ),
    ):
        mda.add_iteration_callback(lambda mda: None)
